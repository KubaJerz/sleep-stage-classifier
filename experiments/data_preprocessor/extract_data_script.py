import os
import mne
import numpy as np
import glob

# Configuration
BASE_DATA_DIR = "/home/kasra/courses/2026-edge-computing/project/dataset/sleepEDF/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"
OUTPUT_DIR = "/home/kasra/courses/2026-edge-computing/project/dataset/sleepEDF/processed/"
CHANNELS = ["EEG Fpz-Cz"]
SAMPLING_RATE = 100

# Mapping labels to integers
LABEL_DICT = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  # Usually N3 and N4 are merged
    "Sleep stage R": 4,
}

def get_file_pairs(data_dir):
    # Find all PSG files
    psg_files = sorted(glob.glob(os.path.join(data_dir, "*PSG.edf")))
    # Find all Hypnogram files
    hypno_files = sorted(glob.glob(os.path.join(data_dir, "*Hypnogram.edf")))
    
    pairs = []
    for psg in psg_files:
        # Extract the ID (e.g., SC4001) to find the matching hypnogram
        file_id = os.path.basename(psg)[:6]
        matching_hypno = [h for h in hypno_files if file_id in h]
        
        if matching_hypno:
            pairs.append((psg, matching_hypno[0]))
            
    return pairs


def process_and_save(pairs):
    # Walk through all subdirectories (sleep-cassette, sleep-telemetry, etc.)
    for psg_path, hypno_path in pairs:
        print(f"Processing: {os.path.basename(psg_path)}")
            
        # Identify Study Type (SC/ST) and Subject ID
        psg_filename = os.path.basename(psg_path)
        study_type = psg_filename[:2].upper() # SC or ST
        subject_id = psg_filename[2:6]
        
        if not os.path.exists(hypno_path):
            print(f"Skipping {psg_filename}: Hypnogram not found.")
            continue

        try:
            # Load Signal
            raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
            raw.pick(CHANNELS)
            raw.resample(SAMPLING_RATE)
            
            # Load Labels
            annot = mne.read_annotations(hypno_path)
            raw.set_annotations(annot, emit_warning=False)

            # Segment into 30s epochs
            events, event_id = mne.events_from_annotations(
                raw, event_id=LABEL_DICT, chunk_duration=30, verbose=False
            )
            
            epochs = mne.Epochs(
                raw, events, event_id, tmin=0., tmax=30. - 1./SAMPLING_RATE, 
                baseline=None, preload=True, verbose=False
            )

            x = epochs.get_data() # (n_epochs, 1, 3000)
            y = epochs.events[:, 2]

            # Setup directory: processed_data/SC/4001/
            save_dir = os.path.join(OUTPUT_DIR, study_type)
            os.makedirs(save_dir, exist_ok=True)
            
            output_file = os.path.join(save_dir, psg_filename.replace("-PSG.edf", ".npz"))
            np.savez(output_file, x=x, y=y)
            
            print(f"Successfully processed: {study_type}/{subject_id} ({len(y)} epochs)")

        except Exception as e:
            print(f"Error processing {psg_filename}: {e}")

if __name__ == "__main__":
    pairs = get_file_pairs(BASE_DATA_DIR)
    process_and_save(pairs)