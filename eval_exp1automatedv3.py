import os
import numpy as np
import sys
import soundfile as sf
import museval
import librosa
import pandas as pd
import re
import argparse
import concurrent.futures
from tqdm import tqdm

# --- Configuration ---
# Adjust this based on your RAM. 
# librosa + museval is heavy. Start with 4. If you have 32GB+ RAM, try 8.
MAX_WORKERS = 4 

# Get the path to your venv's bin folder
venv_bin = os.path.dirname(sys.executable)
os.environ["PATH"] = venv_bin + os.pathsep + os.environ["PATH"]

def load_audio(path):
    """
    Kept exactly as requested: Using librosa.load
    """
    try:
        # Quick check with soundfile to fail fast on corrupt files
        with sf.SoundFile(path) as f:
            pass 
    except RuntimeError as e:
        raise RuntimeError(f"File unreadable or corrupted: {path}\nSoundFile error: {e}")
    
    # Using Librosa as requested
    audio, sr = librosa.load(path, sr=None, mono=False)
    
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    audio = audio.T
    return audio, sr

def evaluate_custom_separation(reference_files, estimate_files):
    refs = []
    for p in reference_files:
        audio, sr = load_audio(p)
        refs.append(audio)
    ests = []
    for p in estimate_files:
        audio, sr = load_audio(p)
        ests.append(audio) 

    min_len = min([x.shape[0] for x in refs + ests])
    refs = [x[:min_len, :] for x in refs]
    ests = [x[:min_len, :] for x in ests]

    references = np.stack(refs)
    estimates = np.stack(ests)

    # Calculate Global SDR (win=infinity)
    scores = museval.evaluate(references, estimates, win=float('inf'))
    return scores

def get_source_files(version_dir):
    """Return sorted list of valid .wav files, ignoring hidden/system files."""
    return sorted([f for f in os.listdir(version_dir)
                   if f.endswith('.wav') and not f.startswith('._') and not f.startswith('.')])

def parse_effect_info(folder_name, original_suffix):
    if folder_name.endswith(original_suffix):
        return None, None
    # Match effect and level, e.g. '_reverb_l2'
    match = re.search(r'_(reverb|delay|compression|bitcrush)_l(\d+)$', folder_name)
    if match:
        return match.group(1), int(match.group(2))
    else:
        return None, None

def parse_song_name(folder_name, original_suffix):
    cleaned_name = re.sub(r'_(reverb|delay|compression|bitcrush)_l\d+$', '', folder_name)
    if cleaned_name.endswith(original_suffix):
        cleaned_name = cleaned_name[:-len(original_suffix)]
    return cleaned_name.strip('_ ').strip()

def process_single_song(song_folder, original_suffix):
    """
    Worker function: Processes ONE song folder containing multiple versions/effects.
    Returns a list of dictionaries (results).
    """
    results = []
    try:
        versions = [d for d in os.listdir(song_folder) if os.path.isdir(os.path.join(song_folder, d))]

        # Identify original version folder
        original_folders = [v for v in versions if v.endswith(original_suffix)]
        if not original_folders:
            return results # Skip
        
        original_folder = original_folders[0]
        original_path = os.path.join(song_folder, original_folder)
        original_files = get_source_files(original_path)

        song_name = parse_song_name(original_folder, original_suffix)

        for v in versions:
            effect, level = parse_effect_info(v, original_suffix)
            if v == original_folder:
                continue
            if effect is None:
                continue

            effect_path = os.path.join(song_folder, v)
            effect_files = get_source_files(effect_path)

            if original_files != effect_files:
                continue

            ref_paths = [os.path.join(original_path, f) for f in original_files]
            est_paths = [os.path.join(effect_path, f) for f in effect_files]

            # Evaluate
            sdr, isr, sir, sar = evaluate_custom_separation(ref_paths, est_paths)
            
            sources = [os.path.splitext(f)[0] for f in original_files]
            for i, source in enumerate(sources):
                results.append({
                    "song": song_name,
                    "effect": effect,
                    "level": level,
                    "source": source,
                    "SDR": np.nanmedian(sdr[i]),
                    "ISR": np.nanmedian(isr[i]),
                    "SIR": np.nanmedian(sir[i]),
                    "SAR": np.nanmedian(sar[i]),
                })
    except Exception as e:
        # Print error but don't crash the whole batch
        print(f"\nError processing {song_folder}: {e}")
        
    return results

def evaluate_all_songs_multicore(root_dir, original_suffix):
    all_results = []
    
    # 1. Collect all folder paths first
    song_folders = []
    for sf in os.listdir(root_dir):
        full_path = os.path.join(root_dir, sf)
        if os.path.isdir(full_path):
            song_folders.append(full_path)
            
    print(f"Found {len(song_folders)} songs. Starting processing with {MAX_WORKERS} cores...")

    # 2. Process in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_song, path, original_suffix): path for path in song_folders}
        
        # Gather results with a progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(song_folders)):
            try:
                result = future.result()
                if result:
                    all_results.extend(result)
            except Exception as e:
                print(f"Wrapper exception: {e}")

    return all_results

class Args:
    # Update these paths to match your current needs
    root_dir = "/Volumes/Ivana/ML Outputs/spleeter_experiment1"
    original_suffix = "_original"
    output = "effect_level_results.xlsx"

def main():
    args = Args()
    
    # Ensure root dir exists
    if not os.path.exists(args.root_dir):
        print(f"Error: Root directory not found: {args.root_dir}")
        return

    results = evaluate_all_songs_multicore(args.root_dir, args.original_suffix)

    if results:
        df = pd.DataFrame(results)
        df.to_excel(args.output, index=False)
        print(f"\nAll results saved to {args.output}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    # Required for multiprocessing on macOS/Windows
    main()