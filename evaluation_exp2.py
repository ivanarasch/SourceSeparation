#import musdb
import museval
import os
import numpy as np
import sys
import librosa



output_dir = os.getcwd()

def load_audio (path):
    audio, sr = librosa.load(path, sr=None, mono=False)
    
    # Librosa loads as (Channels, Samples), but museval needs (Samples, Channels)
    # 1. Handle shapes
    if audio.ndim == 1:
        # If (Samples,), convert to (1, Samples)
        audio = audio[np.newaxis, :]
        
    # 2. Transpose to (Samples, Channels)
    audio = audio.T 
    return audio

def evaluate_custom_separation(reference_files, estimate_files):
    """
    reference_files: list of paths to ground truth files (e.g. ['vocals_true.wav', 'drums_true.wav'])
    estimate_files: list of paths to your separated files (e.g. ['vocals_pred.wav', 'drums_pred.wav'])
    """
    print("Loading files...")
    # 1. Load References into a list of arrays
    refs = []
    for path in reference_files:
        audio = load_audio(path)
        refs.append(audio)
        
    # 2. Load Estimates into a list of arrays
    ests = []
    for path in estimate_files:
        audio = load_audio(path)
        ests.append(audio)
    
    min_len = min([x.shape[0] for x in refs + ests])
    
    # Crop all arrays to min_len
    refs = [x[:min_len, :] for x in refs]
    ests = [x[:min_len, :] for x in ests]

    # 3. Stack them into the shape (n_sources, n_samples, n_channels)
    # Note: All files must have the exact same length and sample rate!
    references = np.stack(refs)
    estimates = np.stack(ests)

    # 4. Run Evaluation
    # This computes SDR, ISR, SIR, SAR
    print("Computing metrics (this may take a moment)...")
    scores = museval.evaluate(references, estimates, win=float('inf'))
    
    return scores

true_paths = ["/Users/Sanjay/Downloads/BKS_-_Too_Much/BKS_-_Too_Much_original/vocals.wav", 
              "/Users/Sanjay/Downloads/BKS_-_Too_Much/BKS_-_Too_Much_original/drums.wav",
              "/Users/Sanjay/Downloads/BKS_-_Too_Much/BKS_-_Too_Much_original/other.wav",
              "/Users/Sanjay/Downloads/BKS_-_Too_Much/BKS_-_Too_Much_original/bass.wav"]
pred_paths = ["/Users/Sanjay/Downloads/BKS_-_Too_Much/BKS_-_Too_Much_reverb_l4/vocals.wav", 
              "/Users/Sanjay/Downloads/BKS_-_Too_Much/BKS_-_Too_Much_reverb_l4/drums.wav", 
              "/Users/Sanjay/Downloads/BKS_-_Too_Much/BKS_-_Too_Much_reverb_l4/other.wav",
              "/Users/Sanjay/Downloads/BKS_-_Too_Much/BKS_-_Too_Much_reverb_l4/bass.wav"]

# Run
try:
    sdr, isr, sir, sar = evaluate_custom_separation(true_paths, pred_paths)

    print("--- Results ---")
    # museval returns scores for every 1-second frame. 
    # We usually take the median to get a single score per source.
    sources = ["Vocals", "Drums", "Other", "Bass"]
    for i, name in enumerate(sources):
        print(f"{name} SDR: {np.nanmedian(sdr[i]):.2f} dB")
        print(f"{name} SIR: {np.nanmedian(sir[i]):.2f} dB")
        print(f"{name} SAR: {np.nanmedian(sar[i]):.2f} dB")

except Exception as e:
    print(f"Error during evaluation: {e}")
