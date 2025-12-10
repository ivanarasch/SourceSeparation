import os
import numpy as np
import sys
import soundfile as sf
# Get the path to your venv's bin folder
# (assuming you are running python FROM that venv)
venv_bin = os.path.dirname(sys.executable)
# Force-add this folder to the system PATH
os.environ["PATH"] = venv_bin + os.pathsep + os.environ["PATH"]
#print("Current PATH for FFMPEG:", os.environ["PATH"])
import museval
import librosa
import pandas as pd
import re
import argparse

def load_audio(path):
    try:
        # Use soundfile just to check if the file is readable and not corrupted
        with sf.SoundFile(path) as f:
            pass  # if this succeeds, file is readable
    except RuntimeError as e:
        raise RuntimeError(f"File unreadable or corrupted: {path}\nSoundFile error: {e}")
    print(f"Loading audio: {path}")
    audio, sr = librosa.load(path, sr=None, mono=False)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    audio = audio.T
    print(f"Loaded {path} successfully.")
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

    scores = museval.evaluate(references, estimates, win=float('inf'))
    return scores

def get_source_files(version_dir):
    """Return sorted list of valid .wav files, ignoring hidden/system files."""
    return sorted([f for f in os.listdir(version_dir)
                   if f.endswith('.wav') and not f.startswith('._') and not f.startswith('.')])

def parse_effect_info(folder_name, original_suffix):
    """
    Extract effect type and level from folder name.
    Returns (effect, level) or (None, None) if original.
    """
    if folder_name.endswith(original_suffix):
        return None, None

    # Match effect and level, e.g. '_reverb_l2'
    match = re.search(r'_(reverb|delay|compression)_l(\d+)$', folder_name)
    if match:
        return match.group(1), int(match.group(2))
    else:
        return None, None

def parse_song_name(folder_name, original_suffix):
    """
    Remove effect suffixes or original suffix from folder name to get song name.
    """
    # Remove effect suffixes like '_reverb_l2', '_delay_l1', '_compression_l4'
    cleaned_name = re.sub(r'_(reverb|delay|compression|bitcrush)_l\d+$', '', folder_name)
    # Remove original suffix
    if cleaned_name.endswith(original_suffix):
        cleaned_name = cleaned_name[:-len(original_suffix)]
    return cleaned_name.strip('_ ').strip()

def evaluate_song_folder(song_folder, original_suffix):
    results = []
    versions = [d for d in os.listdir(song_folder) if os.path.isdir(os.path.join(song_folder, d))]

    # Identify original version folder
    original_folders = [v for v in versions if v.endswith(original_suffix)]
    if not original_folders:
        print(f"No original version found in {song_folder}, skipping.")
        return results
    original_folder = original_folders[0]
    original_path = os.path.join(song_folder, original_folder)
    original_files = get_source_files(original_path)

    song_name = parse_song_name(original_folder, original_suffix)

    for v in versions:
        effect, level = parse_effect_info(v, original_suffix)
        if v == original_folder:
            continue
        if effect is None:
            print(f"Skipping unknown version format: {v}")
            continue

        effect_path = os.path.join(song_folder, v)
        effect_files = get_source_files(effect_path)

        if original_files != effect_files:
            print(f"File mismatch between original and {v} in {song_folder}, skipping.")
            continue

        ref_paths = [os.path.join(original_path, f) for f in original_files]
        est_paths = [os.path.join(effect_path, f) for f in effect_files]

        print(f"Evaluating {song_name} - Effect: {effect} Level: {level}")
        try:
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
            print(f"Error evaluating {v} in {song_folder}: {e}")

    return results

def evaluate_all_songs(root_dir, original_suffix):
    all_results = []
    for song_folder in os.listdir(root_dir):
        full_path = os.path.join(root_dir, song_folder)
        if os.path.isdir(full_path):
            print(f"Processing song folder: {song_folder}")
            results = evaluate_song_folder(full_path, original_suffix)
            all_results.extend(results)
    return all_results

class Args:
    root_dir = "/Volumes/Ivana/ML Outputs/spleeter_experiment2"
    original_suffix = "_original"
    output = "results.xlsx"


def main():

    args = Args()

    #parser = argparse.ArgumentParser(description="Evaluate source separation metrics for songs with effects.")
    #parser.add_argument("root_dir", help="Root directory containing song folders")
    #parser.add_argument("--original_suffix", default="_original",
    #                    help="Suffix for original (reference) versions (default: _original)")
    #parser.add_argument("--output", default="effect_level_source_separation_metrics.xlsx",
    #                    help="Output Excel filename (default: effect_level_source_separation_metrics.xlsx)")

    #args = parser.parse_args()

    results = evaluate_all_songs(args.root_dir, args.original_suffix)

    if results:
        df = pd.DataFrame(results)
        df.to_excel(args.output, index=False)
        print(f"\nAll results saved to {args.output}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()
