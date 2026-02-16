import numpy as np
import soundfile as sf
import parselmouth

# im gonna start putting more comments from now on

def td_psola(
    audio, 
    sr, 
    pitch_semitones=0.0, 
    stretch_factor=1.0, 
    formant_semitones=0.0, 
    voice_drive=0.0, # experiment param
    drive_speed=1.0, # experiment param
    fry_intensity=0.0, # experiment param
    pitch_floor=50.0,
    pitch_ceiling=1200.0
):
    if audio.ndim > 1: audio = audio[:, 0]
    
    pitch_factor = 2.0 ** (pitch_semitones / 12.0)
    formant_factor = 2.0 ** (formant_semitones / 12.0)
    
    # pitch marking
    snd = parselmouth.Sound(audio, sr)
    pitch_track = snd.to_pitch(pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    pulses = parselmouth.praat.call([snd, pitch_track], "To PointProcess (cc)") # usint praat's implementation for proof of concept to get epochs
    num_points = parselmouth.praat.call(pulses, "Get number of points")         # implementation shouldn't be that hard for ChopSampler later
    epochs_sec = [parselmouth.praat.call(pulses, "Get time from index", i) for i in range(1, num_points + 1)]
    praat_epochs = (np.array(epochs_sec) * sr).astype(int)
    
    hop_unvoiced = int(0.01 * sr) 
    
    # build epoch timeline
    full_epochs = []
    if len(praat_epochs) > 0:
        curr = 0
        for ep in praat_epochs:
            while ep - curr > 1.5 * hop_unvoiced:
                full_epochs.append(curr); curr += hop_unvoiced
            full_epochs.append(ep); curr = ep
        while len(audio) - curr > hop_unvoiced:
            full_epochs.append(curr + hop_unvoiced); curr = full_epochs[-1]
    else:
        full_epochs = np.arange(0, len(audio), hop_unvoiced).tolist()
    
    epochs = np.array(full_epochs)
    
    # we treat is_voiced as a float (1.0 or 0.0) so we can interpolate it later
    is_voiced = np.array([1.0 if (not np.isnan(pitch_track.get_value_at_time(e/sr)) and pitch_track.get_value_at_time(e/sr) > 0) else 0.0 for e in epochs])
    T0_array = np.array([sr / pitch_track.get_value_at_time(e/sr) if v == 1.0 else hop_unvoiced for e, v in zip(epochs, is_voiced)])

    # synthesis plan
    target_len = int(len(audio) * stretch_factor)
    out_len = target_len + int(sr * 2.0)
    output = np.zeros(out_len) 

    t_s, drive_phase = 0.0, 0.0
    prev_T_s = hop_unvoiced

    while t_s < target_len:
        t_a = t_s / stretch_factor
        if t_a >= len(audio) - 1: break

        idx1 = np.searchsorted(epochs, t_a) - 1
        idx1 = np.clip(idx1, 0, len(epochs)-2)
        idx2 = idx1 + 1
        
        weight = (t_a - epochs[idx1]) / (epochs[idx2] - epochs[idx1]) if (epochs[idx2] - epochs[idx1]) > 0 else 0.0
        
        # a smooth crossfade value between 0.0 (pure unvoiced) and 1.0 (pure voiced)
        v1, v2 = is_voiced[idx1], is_voiced[idx2]
        voicing_mix = (1.0 - weight) * v1 + weight * v2
        
        T0_interp = (1.0 - weight) * T0_array[idx1] + weight * T0_array[idx2]
        
        # blend the step sizes as well
        T_s_target = voicing_mix * (T0_interp / pitch_factor) + (1.0 - voicing_mix) * hop_unvoiced
        
        max_delta = hop_unvoiced * 0.5 
        if T_s_target > prev_T_s + max_delta: T_s = prev_T_s + max_delta
        elif T_s_target < prev_T_s - max_delta: T_s = prev_T_s - max_delta
        else: T_s = T_s_target
            
        fry_offset, fry_amp = 0, 1.0

        ### processing

        # voiced
        if voicing_mix > 0.0:
            extract_win_size_v = int(np.round(2.0 * T0_interp))
            extract_win_size_v += extract_win_size_v % 2 
            
            def get_aligned_grain(idx, size):
                ep = epochs[idx]
                start = int(ep - size // 2)
                end = start + size
                if start < 0: return np.pad(audio[:end], (-start, 0), mode='constant')
                elif end > len(audio): return np.pad(audio[start:], (0, end - len(audio)), mode='constant')
                return audio[start:end]

            morphed_pulse = (1 - weight) * get_aligned_grain(idx1, extract_win_size_v) + weight * get_aligned_grain(idx2, extract_win_size_v)
            source_rms = np.sqrt(np.mean(morphed_pulse**2)) + 1e-12

            if formant_factor != 1.0:
                orig_idx = np.linspace(0, 1, len(morphed_pulse))
                new_len = int(len(morphed_pulse) / formant_factor)
                new_len += new_len % 2 
                shifted_pulse = np.interp(np.linspace(0, 1, new_len), orig_idx, morphed_pulse)
            else:
                shifted_pulse = morphed_pulse
            
            shifted_pulse *= np.hanning(len(shifted_pulse))
            
            # loundness comp
            current_rms = np.sqrt(np.mean(shifted_pulse**2)) + 1e-12
            density_comp = np.sqrt(max(T_s, 1.0) / max(T0_interp, 1.0)) 
            shifted_pulse *= ((source_rms / current_rms) * density_comp)

            # experimental effects
            if voice_drive > 0:
                drive_phase += (2.0 * np.pi * drive_speed * (T_s / sr))
                shifted_pulse *= (1.0 + (np.sin(drive_phase) * voice_drive))
            if fry_intensity > 0:
                fry_offset = (np.random.randn() * T0_interp * 0.12) * fry_intensity
                if int(t_s / T_s) % 2 == 0: fry_amp = 1.0 - (0.5 * fry_intensity)
                shifted_pulse *= fry_amp

            # apply crossfade weight
            voiced_grain = shifted_pulse * voicing_mix
            
            # overlap-add voiced
            ts_pos = int(t_s + fry_offset)
            start_s = ts_pos - len(voiced_grain) // 2
            end_s = start_s + len(voiced_grain)
            if start_s >= 0 and end_s <= out_len:
                output[start_s:end_s] += voiced_grain
            elif start_s >= 0 and start_s < out_len:
                output[start_s:out_len] += voiced_grain[:out_len - start_s]

        # unvoiced
        if voicing_mix < 1.0:
            unvoiced_weight = 1.0 - voicing_mix

            # determine req grain size
            required_final_size = max(T_s, prev_T_s) * 2.0
            extract_win_size_u = int(np.round(max(2.0 * hop_unvoiced, required_final_size)))
            extract_win_size_u += extract_win_size_u % 2 
            
            # analysis window centered at t_a
            start = int(int(t_a) - extract_win_size_u // 2)
            end = start + extract_win_size_u
            
            # boundary conditions
            if start < 0:
                morphed_pulse_u = np.pad(audio[:end], (-start, 0), mode="constant")
            elif end > len(audio):
                morphed_pulse_u = np.pad(audio[start:], (0, end - len(audio)), mode="constant")
            else:
                morphed_pulse_u = audio[start:end]
            
            #if extraction failed somehow
            if len(morphed_pulse_u) != extract_win_size_u:
                morphed_pulse_u = np.zeros(extract_win_size_u)
            
            unvoiced_grain = (morphed_pulse_u * np.hanning(extract_win_size_u))
            
            # overlap-add unvoiced
            start_s = int(t_s) - len(unvoiced_grain) // 2
            end_s = start_s + len(unvoiced_grain)
            if start_s >= 0 and end_s <= len(output):
                output[start_s:end_s] += unvoiced_grain
            elif start_s >= 0 and start_s < len(output):
                output[start_s:] += unvoiced_grain[:len(output) - start_s]

        # move on by one pitch period at a time and store it for next iter
        t_s += T_s
        prev_T_s = T_s

    # normalization
    orig_max = np.max(np.abs(audio))
    out_max = np.max(np.abs(output))
    if out_max > 0:
        output = (output / out_max) * orig_max
        
    return output[:target_len]

if __name__ == "__main__":
    audio, sr = sf.read("test2.wav")
    result = td_psola(
        audio, sr, 
        pitch_semitones=12, # +/- value to shift, can be decimal numbers
        stretch_factor=1.0, # multiplier
        formant_semitones=0, #same as pitch semitones
        #voice_drive=2.0, # multiplier     
        #drive_speed=75.0, # multiplier   
        #fry_intensity=0.75, # multiplier
    )
    sf.write("test2_+12.wav", result, sr)
    print("Done!")
