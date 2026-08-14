"""
Feature extraction module.
Run python feature_extraction.py to create features.csv file,
that contains features of all patients and associated label (0 healthy, 1 diagnosis).
"""
from pathlib import Path
from multiprocessing import freeze_support

import librosa
import parselmouth
import numpy as np
import pandas as pd
import spkit as sp
from parselmouth.praat import call  # pylint: disable=import-error
from scipy.stats import skew
from tqdm import tqdm

# Disable warnings
import os, warnings
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


# setup seed to ensure reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def round_to_significant(x, n=6) -> float:
    """
    Round the number to n significant digits.
    :param x: number to round
    :param n: number of significant digits. If None, no rounding is done
    :return: rounded number
    """
    x = np.asarray(x, dtype=np.float64)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(n-1))
    mags = 10 ** (n - 1 - np.floor(np.log10(x_positive)))
    result = np.round(x * mags) / mags
    return result


def extract_time_domain_features(raw_data_praat, raw_data_librosa) -> dict:
    """
    Extract time domain features from the voice file.
    :param raw_data_praat: raw data from the voice file (Praat)
    :param raw_data_librosa: raw data from the voice file (librosa)
    :return: dictionary with extracted features
    """
    features = {}

    # pitch object and point process object
    pitch = call(raw_data_praat, "To Pitch", 0.0, 50, 500)
    point_process = call(raw_data_praat, "To PointProcess (periodic, cc)", 50, 500)

    # estimate pitch
    pitch_data = pitch.selected_array['frequency']
    pitch_data[pitch_data == 0] = np.nan

    # pitch difference feature
    diff_pitch = (np.nanmax(pitch_data) - np.nanmin(pitch_data)) / np.nanmin(pitch_data)
    features["diff_pitch"] = diff_pitch

    # get mean_f0 feature
    mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
    features["mean_f0"] = mean_f0

    # standard deviation of mean_f0
    stdev_f0 = call(pitch, "Get standard deviation", 0, 0, "Hertz")
    features["stdev_f0"] = stdev_f0

    # hnr feature
    harmonicity = call(raw_data_praat, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    features["hnr"] = hnr

    # jitter
    local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    features["jitter"] = local_jitter

    #shimmer
    local_shimmer = call([raw_data_praat, point_process],
                         "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    features["shimmer"] = local_shimmer

    # zero crossing rate feature
    zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y=raw_data_librosa)[0])
    features["zero_crossing_rate"] = zero_crossing

    # shannon entropy feature
    shannon_entropy = sp.entropy(raw_data_librosa, alpha=1)
    features["shannon_entropy"] = shannon_entropy

    # raw signal skewness feature
    features["skewness"] = skew(raw_data_librosa)

    return features


def extract_spectral_features(raw_data_librosa, sampling_rate) -> dict:
    """
    Extract spectral features from the voice file.
    :param raw_data_librosa: raw data from the voice file (librosa)
    :param sampling_rate: sampling rate of the voice file
    :return: dictionary with extracted features
    """
    features = {}

    # spectral flatness feature
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=raw_data_librosa), axis=1)
    features["spectral_flatness"] = spectral_flatness[0]

    # spectral roll-off feature
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=raw_data_librosa, sr=sampling_rate), axis=1)
    features["spectral_rolloff"] = spectral_rolloff[0]

    # spectral centroid feature
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=raw_data_librosa, sr=sampling_rate), axis=1)
    features["spectral_centroid"] = spectral_centroid[0]

    # spectral contrast feature
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=raw_data_librosa, sr=sampling_rate), axis=1)
    features["spectral_contrast"] = spectral_contrast

    return features


# pylint: disable=too-many-locals
# pylint: enable=too-many-locals


def extract_features(voice_path: Path, patient_table) -> dict:
    """
    Feature extraction for single patient.
    :param Path voice_path:
    :param pd.DataFrame patient_table: table with patient information
    :return: dict with all utilized features
    """
    session_id = int(voice_path.name.split("-")[0])

    # Saving the session ID and the pathology information
    features = {"session_id": session_id,
                "pathology": patient_table[patient_table.session_id == session_id]["pathology_binary"].values[0],
                "sex": patient_table[patient_table.session_id == session_id]["sex"].values[0],
                "age": patient_table[patient_table.session_id == session_id]["age"].values[0]}

    # Read raw sound data (librosa)
    raw_data_librosa, sampling_rate = librosa.load(str(voice_path), sr=None)
    raw_data_librosa = np.asarray(raw_data_librosa, dtype=np.float64)
    # Read raw sound data (Praat)
    raw_data_praat = parselmouth.Sound(str(voice_path))
    # Extract time domain features
    time_domain_features = extract_time_domain_features(raw_data_praat,raw_data_librosa)
    features.update(time_domain_features)

    # Extract spectral features
    spectral_features = extract_spectral_features(raw_data_librosa, sampling_rate)
    features.update(spectral_features)


    # check NaN
    for key, value in features.items():
        if isinstance(value, (np.ndarray, list)):
            if any(np.isnan(value)):
                features[key] = [0 if np.isnan(x) else x for x in value]
        else:
            if np.isnan(value):
                features[key] = 0.0
    return features


def main(source_dictionary: Path, round_digits = 6):
    """
    Main function for feature extraction.
    :param source_dictionary: path to the folder with the recordings
    :param round_digits: number of significant digits to round the features to. If None, no rounding is done
    :return: None
    """
    file_paths = sorted(list(source_dictionary.glob("*.wav")))

    data_to_dump = []
    # features that need to be rounded to make the results reproducible
    to_round = ["diff_pitch", "mean_f0", "stdev_f0", "hnr", "jitter", "shimmer",
                "spectral_centroid", "spectral_contrast", "spectral_flatness", "spectral_rolloff",
                "zero_crossing_rate", "shannon_entropy", "skewness"]

    # read the information about patients
    patient_table = pd.read_csv(source_dictionary.joinpath("dataset_information.csv"))

    for file in tqdm(file_paths):
         # Extract features for each patient
        res = extract_features(file, patient_table=patient_table)

        # round the features to the specified number of significant digits
        # this is done to make the features creation reproducible
        # as the results may vary slightly depending on the machine
        for key in to_round:
            res[key] = round_to_significant(res[key], round_digits).tolist()
        # append the features to the list
        data_to_dump.append(res)

    # dump feature dataset
    dataset = pd.DataFrame.from_dict(data_to_dump)
    dataset = dataset.sort_values("session_id").reset_index(drop=True)

    dataset.to_csv("features.csv", index=False, lineterminator="\n")


if __name__ == "__main__":
    freeze_support()
    main(Path(".", "dataset"), round_digits=6)
