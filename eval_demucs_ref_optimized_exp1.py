import os
import numpy as np
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

    # 4. Compute Metrics
    scores = museval.evaluate(references, estimates, win=1.0)
    return scores

def find_model_files(model_root, song_name, stem_name):
    """
    Tries to find the model's output files by checking both folder structures.
    
    Structure A (Nested): ModelRoot / SongName / stem_vocals / [vocals.wav, accompaniment.wav]
    Structure B (Flat):   ModelRoot / SongName_stem / [vocals.wav, no_vocals.wav]
    
    Returns: (list_of_file_paths, structure_type_name) or (None, None)
    """
    
    # --- Strategy 1: Nested Structure (Spleeter-style) ---
    # Folder: SongName / bass_vocals
    nested_dir = os.path.join(model_root, song_name, f"{stem_name}_vocals")
    if os.path.exists(nested_dir):
        # Check standard filenames
        v_path = os.path.join(nested_dir, "vocals.wav")
        acc_path = os.path.join(nested_dir, "accompaniment.wav")
        
        # Sometimes Nested uses 'no_vocals.wav' too, check for it
        if not os.path.exists(acc_path):
            acc_path = os.path.join(nested_dir, "no_vocals.wav")
            
        if os.path.exists(v_path) and os.path.exists(acc_path):
            return [v_path, acc_path], "Nested"

    # --- Strategy 2: Flat Structure (Demucs-style) ---
    # Folder: SongName_bass
    flat_dir_name = f"{song_name}_{stem_name}"
    flat_dir = os.path.join(model_root, flat_dir_name)
    
    if os.path.exists(flat_dir):
        # Demucs typically uses 'no_vocals.wav' instead of 'accompaniment.wav'
        v_path = os.path.join(flat_dir, "vocals.wav")
        acc_path = os.path.join(flat_dir, "no_vocals.wav")
        
        # Fallback if named accompaniment.wav
        if not os.path.exists(acc_path):
            acc_path = os.path.join(flat_dir, "accompaniment.wav")

        if os.path.exists(v_path) and os.path.exists(acc_path):
            return [v_path, acc_path], "Flat"

    return None, None

def evaluate_song(model_name, model_root, ref_root, song_folder):
    results = []
    
    # Path to specific song in Reference (Ground Truth is assumed to always be Nested)
    ref_song_path = os.path.join(ref_root, song_folder)

    # Identify stems based on Reference folders (e.g. bass_vocals, drums_vocals)
    # We iterate the REFERENCE folders to know what to look for in the Model
    if not os.path.exists(ref_song_path):
        return results

    subfolders = [d for d in os.listdir(ref_song_path) 
                  if os.path.isdir(os.path.join(ref_song_path, d))]

    for sub in subfolders:
        # Expecting folders like "bass_vocals", "drums_vocals"
        if not sub.endswith("_vocals"):
            continue
            
        # Extract "bass", "drums", "accompaniment"
        stem_name = sub.replace("_vocals", "")
        
        # 1. Get Reference Files
        ref_dir = os.path.join(ref_song_path, sub)
        ref_files = [
            os.path.join(ref_dir, "vocals.wav"),
            os.path.join(ref_dir, "accompaniment.wav") # Assuming ref uses this name
        ]
        
        if not all(os.path.exists(f) for f in ref_files):
            # Check for alternative ref naming
            ref_files[1] = os.path.join(ref_dir, "no_vocals.wav")
            if not all(os.path.exists(f) for f in ref_files):
                continue

        # 2. Find Corresponding Model Files (Auto-detect structure)
        est_files, struct_type = find_model_files(model_root, song_folder, stem_name)
        
        if not est_files:
            print(f"   [Missing] Could not find {stem_name} for {song_folder} in {model_name}")
            continue

        print(f"   Evaluating: {song_folder} | {stem_name} (Found {struct_type} style)")

        try:
            # Source 0 = Vocals, Source 1 = The Instrument (Bass/Drums/etc)
            sdr, isr, sir, sar = evaluate_pair(ref_files, est_files)
            
            source_labels = ["vocals", stem_name]
            
            for i, label in enumerate(source_labels):
                results.append({
                    "Model": model_name,
                    "Song": song_folder,
                    "Combination": sub,       # e.g. bass_vocals
                    "Source": label,          # e.g. bass
                    "SDR": np.nanmedian(sdr[i]),
                    "SIR": np.nanmedian(sir[i]),
                    "SAR": np.nanmedian(sar[i]),
                    "ISR": np.nanmedian(isr[i])
                })

        except Exception as e:
            print(f"   [Error] Evaluation failed: {e}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Universal Source Separation Evaluation.")
    
    parser.add_argument("--ref", required=True, help="Path to Reference (Ground Truth) Root")
    parser.add_argument("--models", nargs='+', required=True, help="List of Model Root Directories")
    parser.add_argument("--output", default="comparison_results.xlsx", help="Output Excel filename")

    args = parser.parse_args()

    all_results = []
    print(f"Reference Path: {args.ref}")

    # We iterate over the SONGS in the REFERENCE folder first
    # This ensures we only look for songs that actually exist in the ground truth
    ref_songs = [d for d in os.listdir(args.ref) if os.path.isdir(os.path.join(args.ref, d))]
    
    for model_path in args.models:
        model_name = os.path.basename(os.path.normpath(model_path))
        print(f"\n--- Processing Model: {model_name} ---")
        
        if not os.path.exists(model_path):
            print(f"Error: Path not found {model_path}")
            continue

        for song in ref_songs:
            # Skip hidden folders
            if song.startswith("."): continue
            
            results = evaluate_song(model_name, model_path, args.ref, song)
            all_results.extend(results)

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_excel(args.output, index=False)
        print(f"\nSuccess! Saved to {args.output}")
        # Print valid SDR summary
        print(df.groupby(["Model", "Source"])["SDR"].median())
    else:
        print("\nNo results generated. Check paths.")

if __name__ == "__main__":
    main()
