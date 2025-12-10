import os
import numpy as np
import sys
import soundfile as sf
import museval
import librosa
import pandas as pd
import argparse
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def load_audio(path):
    """
    Robust audio loading using Librosa. 
    Forces stereo (2 channels) and original sample rate.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        # Use soundfile just to check if the file is readable
        with sf.SoundFile(path) as f:
            pass
    except Exception as e:
        raise RuntimeError(f"File unreadable: {path}\nError: {e}")

    # Load with librosa (sr=None ensures no resampling)
    audio, sr = librosa.load(path, sr=None, mono=False)
    
    # Handle shape (Channels, Samples) -> (Samples, Channels)
    if audio.ndim == 1:
        # If mono, add channel dimension: (Samples,) -> (1, Samples)
        audio = audio[np.newaxis, :]
    
    # Transpose to (Samples, Channels)
    audio = audio.T
    return audio

def evaluate_pair(reference_files, estimate_files):
    """
    Calculates BSSEval metrics for a pair of sources (e.g. Vocals + Accompaniment).
    """
    print(f"   Loading {len(estimate_files)} stems...")
    
    # 1. Load Audio
    refs = [load_audio(p) for p in reference_files]
    ests = [load_audio(p) for p in estimate_files]

    # 2. Align Lengths (Crop to shortest)
    min_len = min([x.shape[0] for x in refs + ests])
    refs = [x[:min_len, :] for x in refs]
    ests = [x[:min_len, :] for x in ests]

    # 3. Stack for Museval (n_sources, n_samples, n_channels)
    references = np.stack(refs)
    estimates = np.stack(ests)

    # 4. Compute Metrics
    # win=1.0 uses standard 1-second framing (recommended)
    # win=float('inf') uses global scoring (faster, but less detailed)
    scores = museval.evaluate(references, estimates, win=1.0)
    return scores

def get_instrument_name_from_folder(folder_name):
    """
    Infers the instrument name from the folder 'instrument_vocals'.
    e.g., 'bass_vocals' -> 'bass'
          'accompaniment_vocals' -> 'accompaniment'
    """
    # Split by '_vocals' (assuming folder structure is always 'something_vocals')
    if "_vocals" in folder_name:
        return folder_name.replace("_vocals", "")
    return "accompaniment" # Default fallback

def evaluate_song_folder(model_name, model_root, ref_root, song_folder):
    results = []
    
    # Path to specific song in Model and Reference
    model_song_path = os.path.join(model_root, song_folder)
    ref_song_path = os.path.join(ref_root, song_folder)

    if not os.path.exists(ref_song_path):
        print(f"   [Skipping] Song '{song_folder}' not found in Reference directory.")
        return results

    # Iterate over combinations (bass_vocals, drums_vocals, etc.)
    combinations = [d for d in os.listdir(model_song_path) 
                   if os.path.isdir(os.path.join(model_song_path, d))]

    for combo in combinations:
        model_combo_path = os.path.join(model_song_path, combo)
        ref_combo_path = os.path.join(ref_song_path, combo)

        if not os.path.exists(ref_combo_path):
            print(f"   [Skipping] Combination '{combo}' missing in Reference.")
            continue

        # Define file targets based on your screenshot
        # We expect: 'vocals.wav' and 'accompaniment.wav'
        targets = ['vocals.wav', 'accompaniment.wav']
        
        ref_files = [os.path.join(ref_combo_path, t) for t in targets]
        est_files = [os.path.join(model_combo_path, t) for t in targets]

        # Check if all files exist
        if not all(os.path.exists(f) for f in ref_files + est_files):
            print(f"   [Warning] Missing .wav files in {combo}. Skipping.")
            continue

        # Determine Source Names for logging
        # Target 0 is always Vocals
        # Target 1 is the 'Instrument' defined by the folder name
        inst_name = get_instrument_name_from_folder(combo)
        source_names = ["vocals", inst_name]

        print(f"   Evaluating: {song_folder} | {combo}")
        
        try:
            sdr, isr, sir, sar = evaluate_pair(ref_files, est_files)
            
            # Append results for both sources
            for i, source in enumerate(source_names):
                results.append({
                    "Model": model_name,
                    "Song": song_folder,
                    "Combination": combo,   # e.g., bass_vocals
                    "Source": source,       # e.g., bass
                    "SDR": np.nanmedian(sdr[i]),
                    "SIR": np.nanmedian(sir[i]),
                    "SAR": np.nanmedian(sar[i]),
                    "ISR": np.nanmedian(isr[i])
                })

        except Exception as e:
            print(f"   [Error] Failed to evaluate {song_folder}/{combo}: {e}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Compare Source Separation Models.")
    
    # 1. Reference Directory (Ground Truth)
    parser.add_argument("--ref", required=True, help="Path to the Ground Truth (Reference) Root Directory")
    
    # 2. Output File
    parser.add_argument("--output", default="model_comparison_results.xlsx", help="Output Excel filename")

    # 3. Model Directories (Accepts multiple)
    # Usage: --models "Path/To/ModelA" "Path/To/ModelB"
    parser.add_argument("--models", nargs='+', required=True, help="List of paths to Model Output Root Directories")

    args = parser.parse_args()

    all_results = []

    print(f"Reference Path: {args.ref}")
    
    # Iterate over each model provided
    for model_path in args.models:
        model_name = os.path.basename(os.path.normpath(model_path)) # Use folder name as Model Name
        print(f"\n--- Processing Model: {model_name} ---")
        
        if not os.path.exists(model_path):
            print(f"Error: Model path not found: {model_path}")
            continue

        # Iterate over songs in the model folder
        songs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
        
        for song in songs:
            results = evaluate_song_folder(model_name, model_path, args.ref, song)
            all_results.extend(results)

    # Save to Excel
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_excel(args.output, index=False)
        print(f"\nSuccess! Results saved to: {args.output}")
        print(df.groupby(["Model", "Source"])["SDR"].median()) # Print a quick summary
    else:
        print("\nNo results computed. Check your folder paths.")

if __name__ == "__main__":
    main()
