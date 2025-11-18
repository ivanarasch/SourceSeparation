from pydub import AudioSegment
import numpy as np
import soundfile as sf

# Convert numpy array to AudioSegment
def audiosegment_from_numpy(samples: np.ndarray, rate: int) -> AudioSegment:
    samples_int16 = (samples * 32767).astype(np.int16)
    audio = AudioSegment(
        samples_int16.tobytes(),
        frame_rate=rate,
        sample_width=2,
        channels=samples.shape[1] if samples.ndim > 1 else 1
    )
    return audio

# Convert AudioSegment to numpy array
def audiosegment_to_numpy(audio: AudioSegment) -> np.ndarray:
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32767
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels))
    return samples

# Reverb effect (slightly more pronounced)
def add_reverb(audio: AudioSegment, delay_ms=100, repeats=5) -> AudioSegment:
    output = audio
    for i in range(1, repeats + 1):
        delayed = AudioSegment.silent(duration=delay_ms * i) + (audio - 6)
        output = output.overlay(delayed)
    return output

# Delay effect (echo with feedback)
def add_delay(audio: AudioSegment, delay_ms=400, repeats=4) -> AudioSegment:
    output = audio
    for i in range(1, repeats + 1):
        delayed = AudioSegment.silent(duration=delay_ms * i) + (audio - 8)
        output = output.overlay(delayed)
    return output

# Aggressive bit-crush distortion
def add_bitcrush(audio: AudioSegment, bit_depth=4, downsample_factor=4) -> AudioSegment:
    samples = audiosegment_to_numpy(audio)
    # Reduce bit depth
    levels = 2 ** bit_depth - 1
    crushed = np.round(samples * levels) / levels
    # Downsample to create aliasing artifacts
    crushed[::downsample_factor] = crushed[::downsample_factor]
    crushed = np.clip(crushed, -1.0, 1.0)
    return audiosegment_from_numpy(crushed, audio.frame_rate)

# Save vocals with effect applied fully (wet)
def save_vocals_only(vocals: np.ndarray, rate: int, effect_func, filename: str):
    vocal_audio = audiosegment_from_numpy(vocals, rate)
    vocal_fx = effect_func(vocal_audio)  # full effect applied
    vocals_fx_np = audiosegment_to_numpy(vocal_fx)
    sf.write(filename, vocals_fx_np, rate)


# Save full mixture with only vocals affected
def save_full_mix_with_vocals_fx(full_mix: np.ndarray, vocals: np.ndarray, rate: int, effect_func, filename: str):
    vocal_audio = audiosegment_from_numpy(vocals, rate)
    vocal_fx = effect_func(vocal_audio)
    vocals_fx_np = audiosegment_to_numpy(vocal_fx)
    
    # Make sure lengths match
    min_len = min(full_mix.shape[0], vocals_fx_np.shape[0])
    output = full_mix.copy()
    # Replace vocals with effected vocals
    output[:min_len] += vocals_fx_np[:min_len] - vocals[:min_len]
    output = np.clip(output, -1.0, 1.0)
    
    sf.write(filename, output, rate)


