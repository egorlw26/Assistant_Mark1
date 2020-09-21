import librosa 
import os
import json

# folder with all audio files in separate folders
DATASET_PATH = "dataset"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050 # 1 second worth of sound in librosa

def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):

    # data dict
    data = {
        "mappings": [], # for the names, like ["on", "off" ...]
        "labels": [], # for the values, like [0, 0, 1, 1, ...]
        "MFCCs" : [],
        "files" : [] # for the files ["dataset/on/1.wav"...]
    }
    
    for i, (dirpath, _, filenames) in enumerate(os.walk(dataset_path)):

        if dirpath is not dataset_path:

            print(dirpath)
            category = dirpath.split("\\")[-1]
            data["mappings"].append(category)

            print (f"Processing {category}")

            for f in filenames:
                filepath = os.path.join(dirpath, f)
                signal, __ = librosa.load(filepath)

                if len(signal) >= SAMPLES_TO_CONSIDER:
                    # We wanna have only 1 second of audio, no longer
                    signal = signal[:SAMPLES_TO_CONSIDER]
                    MFCC = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, hop_length=hop_length, 
                                                n_fft=n_fft)

                    # filling the data
                    data["labels"].append(i-1)
                    data["MFCCs"].append(MFCC.T.tolist())
                    data["files"].append(filepath)

                    print (f"{filepath}:{i-1}")
    
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, JSON_PATH)


