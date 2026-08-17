from warnings import warn

from .record import Record


def only_first_each(records: list[Record]) -> list[Record]:
    """Return records from each speaker's lowest recording ID per pathology class.
    For each speaker lowest recording ID is also the oldest.

    :param records: Records to select from.
    :return: Records belonging to each speaker's oldest recording ID for each pathology class.
    """
    oldest_recording_ids = {}
    for record in records:
        speaker_pathology = (record.patient_id, record.pathology)
        if speaker_pathology not in oldest_recording_ids:
            oldest_recording_ids[speaker_pathology] = record.recording_id
        else:
            oldest_recording_ids[speaker_pathology] = min(
                oldest_recording_ids[speaker_pathology],
                record.recording_id,
            )
    return [
        record
        for record in records
        if record.recording_id
        == oldest_recording_ids[(record.patient_id, record.pathology)]
    ]


def singers_to_healthy(records: list[Record]) -> list[Record]:
    """Relabel singing-voice records as healthy in place and return the records.

    :param records: Records whose pathology labels should be overridden.
    :return: The input records with singing-voice labels changed to healthy labels.
    """

    found_once = False
    for record in records:
        if record.pathologies_str in ["Sängerstimme", "Singing voice"]:
            record.pathology = 0
            record.pathologies_str = ""
            found_once |= True

    if not found_once:
        warn("Run singers_to_healthy but it did not changed anything")

    return records
