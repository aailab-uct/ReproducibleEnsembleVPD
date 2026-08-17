from dataclasses import dataclass

import numpy as np


@dataclass
class Record:
    """Represent one voice recording and its metadata.

    :ivar datbase: Source database name, such as ``SVD``.
    :ivar recording_id: Recording identifier within the source database.
    :ivar vowel: Recorded vowel or phrase identifier.
    :ivar pathology: Health class, where ``0`` is healthy and ``1`` is pathological.
    :ivar pathologies_str: Original pathology label from the source database.
    :ivar patient_id: Source database name combined with the speaker identifier.
    :ivar sex: Speaker sex, where ``0`` is male and ``1`` is female.
    :ivar age: Speaker age in years at the time of recording.
    :ivar signal: Normalized audio signal.
    :ivar fs: Sampling frequency in hertz.
    :ivar num_from_recording: Index of an augmented or segmented instance of the recording.
    :ivar kwargs: Additional source metadata.
    """

    datbase: str
    recording_id: int
    vowel: str
    pathology: int  # 0 - HC, 1 - pathological
    pathologies_str: str
    patient_id: str  # database + speaker id
    sex: int  # 0 - male, 1 - female
    age: int
    signal: np.ndarray
    fs: int
    num_from_recording: int = 0
    kwargs: dict | None = None
