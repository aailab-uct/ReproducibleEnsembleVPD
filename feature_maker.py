"""
expected csv columns
id,database,recording_id,vowel,num_from_recording,pathology(0/1),groups,sex(0/1),age,feature1,...,featureN,notes

num_from_recording is used when raw waveform augmentation or segmentation is used and can be utilized
so script can for example use only one instance for testing. For augmentation 0 is always no augmentation, and this is allways returned.

notes should contain all infromation from the database, such as exact pathology, notes, smoker status...
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import parselmouth
import seaborn as sns
import spkit as sp
from matplotlib import pyplot as plt
from parselmouth.praat import call
from scipy.stats import skew
from tqdm import tqdm

from src.misc import only_first_each, singers_to_healthy
from src.record import Record
from src.svd import get_data as get_svd_data


def exclude_records(
    records: list[Record],
    recording_ids: set[int],
    pathologies: set[str],
    soft: bool = True,
) -> list[Record]:
    """Exclude records matching a recording ID or exact pathology label.

    :param records: Records to filter.
    :param recording_ids: Recording IDs to exclude.
    :param pathologies: Pathology labels to exclude.
    :param soft: Whether pathology exclusions must match the complete label exactly.
    :return: Records that do not match the configured exclusions.
    """
    return [
        record
        for record in records
        if (
            record.recording_id not in recording_ids  # filter out ids
            and (
                not all(
                    pathology in record.pathologies_str for pathology in pathologies
                )
                or soft
            )  # exclude if pathologies_str contains any of the pathologies
            and record.pathologies_str
            not in pathologies  # exclude if pathologies_str fully match any record from the pathologies
        )
    ]


def load_records() -> list[Record]:
    """Load records from every database configured in the YAML file."""
    records = []

    records = get_svd_data(vowels=("a_n",), sexes=("f", "m"))

    # exclude always
    recording_ids = {
        1573,  # corrupted found by us
        87,  # corrupted found by us
        1372,  # corrupted per remarks - beeping or background noise
        2088,  # corrupted per remarks - beeping or background noise
        2077,  # corrupted per remarks - beeping or background noise
        2070,  # corrupted per remarks - beeping or background noise
        2081,  # corrupted per remarks - beeping or background noise
        2426,  # corrupted per remarks - beeping or background noise
        2428,  # corrupted per remarks - beeping or background noise
        2448,  # corrupted per remarks - beeping or background noise
        2450,  # corrupted per remarks - beeping or background noise
        2451,  # corrupted per remarks - beeping or background noise
        2457,  # corrupted per remarks - beeping or background noise
    }
    pathologies = {"Gesangsstimme"}
    records = exclude_records(records, recording_ids, pathologies, soft=False)

    # exclude only if given pathology is only one
    irrelevant_pathologies = {
        "Dysodie",
        "Mutatio",
        "Mutationsfistelstimme",
        "Non-fluency-Syndrom",
        "Poltersyndrom",
        "Sigmatismus",
        "Balbuties",
    }
    records = exclude_records(records, set(), irrelevant_pathologies)

    # apply overrides
    records = [record for record in records if record.age >= 18]  # adult only
    records = singers_to_healthy(records)
    records = only_first_each(records)

    # apply postprocessing
    for record in records:
        record.signal, _ = librosa.effects.trim(
            record.signal, top_db=15, frame_length=256, hop_length=65
        )

    return records


def _extract_record_features(record: Record) -> dict[str, Any] | None:
    """Extract the acoustic features consumed by datasets_generator."""
    signal = record.signal

    f_min, f_max = [
        (75.0, 300.0),  # male
        (100.0, 600.0),  # female
    ][record.sex]

    features = {}

    sound = parselmouth.Sound(signal, sampling_frequency=record.fs)
    pitch = call(sound, "To Pitch", 0.0, f_min, f_max)

    pitch_data = pitch.selected_array["frequency"]
    pitch_data = pitch_data[pitch_data > 0]
    if pitch_data.size == 0:
        return None

    features["mean_f0"] = np.mean(pitch_data)
    features["stdev_f0"] = np.std(pitch_data)
    features["diff_pitch"] = (np.max(pitch_data) - np.min(pitch_data)) / np.min(
        pitch_data
    )

    point_process = call(
        sound,
        "To PointProcess (periodic, cc)",
        f_min,
        f_max,
    )
    features["jitter"] = call(
        point_process,
        "Get jitter (local)",
        0,
        0,
        0.0001,
        0.02,
        1.3,
    )
    features["shimmer"] = call(
        [sound, point_process],
        "Get shimmer (local)",
        0,
        0,
        0.0001,
        0.02,
        1.3,
        1.6,
    )

    harmonicity = call(
        sound,
        "To Harmonicity (cc)",
        0.01,
        f_min,
        0.1,
        1.0,
    )
    features["hnr"] = np.mean(harmonicity)

    features["spectral_contrast"] = np.mean(
        librosa.feature.spectral_contrast(y=signal, sr=record.fs),
        axis=1,
    ).tolist()

    features["spectral_centroid"] = np.mean(
        librosa.feature.spectral_centroid(y=signal, sr=record.fs).squeeze()
    )
    features["spectral_flatness"] = np.mean(
        librosa.feature.spectral_flatness(y=signal).squeeze()
    )
    features["spectral_rolloff"] = np.mean(
        librosa.feature.spectral_rolloff(y=signal, sr=record.fs).squeeze()
    )

    features["zero_crossing_rate"] = np.mean(
        librosa.feature.zero_crossing_rate(y=signal).squeeze()
    )
    features["skewness"] = skew(signal)
    features["shannon_entropy"] = sp.entropy(
        signal,
        alpha=1,  # TODO: reference - stejne jak v JoV
    )

    return features


def feature_extraction() -> Callable[[list[Record]], pd.DataFrame]:
    """Create a pipeline for extracting dataset-generator-compatible features."""

    def extract(records: list[Record]) -> pd.DataFrame:
        rows = []
        for record in tqdm(records):
            record_features = _extract_record_features(record)
            if record_features is None:
                tqdm.write(
                    f"Dropping recording {record.datbase}:{record.recording_id}:{record.vowel}: F0 could not be extracted."
                )
                continue
            row = {
                "database": record.datbase,
                "recording_id": record.recording_id,
                "vowel": record.vowel,
                "num_from_recording": record.num_from_recording,
                "pathology": record.pathology,
                "patient_id": record.patient_id,
                "pathologies_str": record.pathologies_str,
                "sex": record.sex,
                "age": record.age,
            }
            row.update(record_features)
            row["notes"] = record.kwargs
            rows.append(row)

        return pd.DataFrame(rows)

    return extract


def main() -> None:
    """Load and process records described by the selected configuration."""

    # Loading data
    records = load_records()

    # feature extraction
    feature_extraction_pipe = feature_extraction()
    features = feature_extraction_pipe(records)

    features.to_csv(Path("features.csv"))

    # data visualization
    features["Sex"] = features["sex"].map({0: "Male", 1: "Female"})
    features["Status"] = features["pathology"].map({0: "Healthy", 1: "Pathological"})
    print(features.head())
    g= sns.displot(
        features,
        row="Status",
        kind="hist",
        x="age",
        hue="Sex",
        facet_kws={"sharex": True, "sharey": False},
    )
    g.set_titles(template="{row_name}")
    g.set(xlabel="Age [years]")
    plt.savefig("age_histogram_stacked.pdf")
    plt.show()


if __name__ == "__main__":
    main()
