import os
import numpy as np
import soundfile as sf
import museval
import librosa
import pandas as pd
import argparse
import warnings
import concurrent.futures
from tqdm import tqdm

# --- CONFIGURATION ---
# SAFE MODE: Set to 2.
# If you have 32GB+ RAM, you can try 4 or 6.
MAX_WORKERS = 2 

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
        audio = audio[np.newaxis, :]
    
    # Transpose to (Samples, Channels)
    audio = audio.T
    return audio

def evaluate_pair(reference_files, estimate_files):
    """
    Calculates BSSEval metrics for a pair of sources.
    """
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

    # 4. Compute Metrics (Global SDR for speed/stability)
    # Using win=float('inf') is safer for multiprocessing memory usage than win=1.0
    scores = museval.evaluate(references, estimates, win=1.0)
    return scores

def find_model_files(model_root, song_name, stem_name):
    """
    Tries to find the model's output files by checking both Nested and Flat structures.
    """
    # Strategy 1: Nested Structure (Spleeter-style)
    nested_dir = os.path.join(model_root, song_name, f"{stem_name}_vocals")
    if os.path.exists(nested_dir):
        v_path = os.path.join(nested_dir, "vocals.wav")
        acc_path = os.path.join(nested_dir, "accompaniment.wav")
        if not os.path.exists(acc_path):
            acc_path = os.path.join(nested_dir, "no_vocals.wav")
            
        if os.path.exists(v_path) and os.path.exists(acc_path):
            return [v_path, acc_path], "Nested"

    # Strategy 2: Flat Structure (Demucs-style)
    flat_dir_name = f"{song_name}_{stem_name}"
    flat_dir = os.path.join(model_root, flat_dir_name)
    if os.path.exists(flat_dir):
        v_path = os.path.join(flat_dir, "vocals.wav")
        acc_path = os.path.join(flat_dir, "no_vocals.wav")
        if not os.path.exists(acc_path):
            acc_path = os.path.join(flat_dir, "accompaniment.wav")

        if os.path.exists(v_path) and os.path.exists(acc_path):
            return [v_path, acc_path], "Flat"

    return None, None

def process_single_song(args):
    """
    Worker function.
    Args must be packed into a tuple because ProcessPoolExecutor maps 1 argument.
    args: (model_name, model_root, ref_root, song_folder)
    """
    model_name, model_root, ref_root, song_folder = args
    results = []
    
    try:
        ref_song_path = os.path.join(ref_root, song_folder)
        if not os.path.exists(ref_song_path):
            return []

        # Find subfolders (tasks) in the Reference directory
        subfolders = [d for d in os.listdir(ref_song_path) 
                      if os.path.isdir(os.path.join(ref_song_path, d))]

        for sub in subfolders:
            if not sub.endswith("_vocals"):
                continue
                
            stem_name = sub.replace("_vocals", "")
            
            # 1. Get Reference Files
            ref_dir = os.path.join(ref_song_path, sub)
            ref_files = [
                os.path.join(ref_dir, "vocals.wav"),
                os.path.join(ref_dir, "accompaniment.wav")
            ]
            
            # Fallback check for ref filenames
            if not all(os.path.exists(f) for f in ref_files):
                ref_files[1] = os.path.join(ref_dir, "no_vocals.wav")
                if not all(os.path.exists(f) for f in ref_files):
                    continue

            # 2. Find Model Files
            est_files, struct_type = find_model_files(model_root, song_folder, stem_name)
            
            if not est_files:
                continue

            # 3. Evaluate
            sdr, isr, sir, sar = evaluate_pair(ref_files, est_files)
            
            source_labels = ["vocals", stem_name]
            
            for i, label in enumerate(source_labels):
                results.append({
                    "Model": model_name,
                    "Song": song_folder,
                    "Combination": sub,
                    "Source": label,
                    "SDR": np.nanmedian(sdr[i]),
                    "SIR": np.nanmedian(sir[i]),
                    "SAR": np.nanmedian(sar[i]),
                    "ISR": np.nanmedian(isr[i])
                })
                
    except Exception as e:
        # Return empty list on error to keep processing other songs
        # print(f"Error processing {song_folder}: {e}") 
        pass

    return results

def main():
    parser = argparse.ArgumentParser(description="Multicore Universal Source Separation Evaluation.")
    parser.add_argument("--ref", required=True, help="Path to Reference (Ground Truth) Root")
    parser.add_argument("--models", nargs='+', required=True, help="List of Model Root Directories")
    parser.add_argument("--output", default="comparison_results.xlsx", help="Output Excel filename")

    args = parser.parse_args()

    all_results = []
    print(f"Reference Path: {args.ref}")
    print(f"Worker Threads: {MAX_WORKERS}")

    # Gather list of songs from Reference
    ref_songs = [d for d in os.listdir(args.ref) if os.path.isdir(os.path.join(args.ref, d)) and not d.startswith(".")]
    
    for model_path in args.models:
        model_name = os.path.basename(os.path.normpath(model_path))
        print(f"\n--- Processing Model: {model_name} ---")
        
        if not os.path.exists(model_path):
            print(f"Error: Path not found {model_path}")
            continue

        # Prepare tasks for Multiprocessing
        # We create a list of argument tuples: [(model, path, ref, song1), (model, path, ref, song2), ...]
        tasks = [(model_name, model_path, args.ref, song) for song in ref_songs]

        # Execute in Parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # map returns results in order
            results_generator = list(tqdm(executor.map(process_single_song, tasks), total=len(tasks)))
            
            # Flatten the list of lists
            for res in results_generator:
                if res:
                    all_results.extend(res)

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_excel(args.output, index=False)
        print(f"\nSuccess! Saved to {args.output}")
    else:
        print("\nNo results generated. Check paths.")

if __name__ == "__main__":
    main()
