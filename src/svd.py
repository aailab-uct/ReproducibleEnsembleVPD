from pathlib import Path

import nspfile
import numpy as np
import pandas as pd
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta

from .record import Record

SVD_FOLDER = Path("C:\\Users\\jirsat\\Code\\voice_pipeline2\\data") / "SVD"
CSV_FOLDER = Path("C:\\Users\\jirsat\\Code\\voice_pipeline2\\data") / "voice_data.csv"
NSP_INT16_SCALE = 2**15


def get_data(
    vowels: tuple[str] | None = ("a_n",), sexes: tuple = ("f", "m")
) -> list[Record]:
    """Load SVD recordings and their metadata.

    :param vowels: Vowel or phrase identifiers to include.
    :param sexes: Speaker sexes to include.
    :return: Records loaded from matching NSP files.
    """
    voice_data = pd.read_csv(CSV_FOLDER)

    records = []
    for file in SVD_FOLDER.rglob("*.nsp"):
        recording_id, vowel = file.stem.split("-")
        recording_id = int(recording_id)

        # continue only with desired vowels
        if vowels is None or vowel not in vowels:
            continue
        if vowel == "phrase":
            vowel += "_gutten_morgen"  # if other databases contain other phrases this would specify the exact phrase

        # check if we have metadata for the recording
        if recording_id not in voice_data.AufnahmeID.values:
            print(f"Recording {recording_id} is not in metadata file!")  # type: ignore
            # 4 recordings of shape (0, 1) - probably problem in database
            continue
        else:
            metadata: dict = (
                voice_data[voice_data.AufnahmeID == recording_id].iloc[0].to_dict()
            )

        pathology = int(
            metadata["AufnahmeTyp"] == "p"
        )  # equivalent to testing if Pathologien is nan
        # note that Diagnose might be nan even if Pathologien is not

        # parse sex, patient_id and agwe
        sex = (
            1 if metadata["Geschlecht"] == "w" else 0
        )  # there are no other symbols than w and m in this column
        patient_id: int = metadata["SprecherID"]
        age = relativedelta(
            parse(metadata["AufnahmeDatum"]),  # date of recording
            parse(metadata["Geburtsdatum"]),  # birthday
        ).years

        # skip undesired sex
        if not ["m", "f"][sex] in sexes:
            continue

        # add pathology name
        if not pd.isna(metadata["Pathologien"]):
            pathologies_str = metadata["Pathologien"]
        else:
            pathologies_str = ""

        # check if we are able to extract the file
        try:
            fs: int
            signal: np.ndarray
            fs, signal = nspfile.read(
                str(file),  # type: ignore - wrong typing in library
                return_note=False,
            )  # no note is in any of the recordings
            signal = signal.squeeze().astype(np.float64) / NSP_INT16_SCALE
        except ValueError as e:
            print(file, ":", e, "File not included.")
            continue
        except RuntimeError as e:
            print(e, "File not included.")
            continue

        # load remarks if they exist
        remarks_file = file.parents[1] / "remarks" / f"{recording_id}-remarks.txt"
        kwargs = {}
        if remarks_file.exists():
            with open(remarks_file, "r") as f:
                kwargs["remarks"] = ";".join(
                    [line.rstrip() for line in f if line != ""]
                )

        # add diagnose to kwargs if not empty
        if not pd.isna(metadata["Diagnose"]):
            kwargs["Diagnosis"] = metadata["Diagnose"]

        records.append(
            Record(
                datbase="SVD",
                recording_id=recording_id,
                vowel=vowel,
                pathology=pathology,
                patient_id=f"SVD{patient_id}",
                sex=sex,
                age=age,
                signal=signal,
                fs=fs,
                pathologies_str=pathologies_str,
                kwargs=kwargs,
            )
        )

    return records
