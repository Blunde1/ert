import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any
import ert
import ert3

_SOURCE_SEPARATOR = "."


def _prepare_export_parameters(
    workspace: Path,
    experiment_name: str,
    ensemble: ert3.config.EnsembleConfig,
    ensemble_size: int,
) -> Dict[str, List[ert.data.record_data]]:
    inputs = defaultdict(list)
    for input_record in ensemble.input:
        record_name = input_record.record
        record_source = input_record.source.split(_SOURCE_SEPARATOR, maxsplit=1)
        assert len(record_source) == 2

        if record_source[0] == "storage" or record_source[0] == "stochastic":
            exp_name = None if record_source[0] == "storage" else experiment_name
            source = record_source[1] if record_source[0] == "storage" else None
            records_url = ert.storage.get_records_url(workspace, exp_name)
            future = ert.storage.get_record_storage_transmitters(
                records_url=records_url,
                record_name=record_name,
                record_source=source,
                ensemble_size=ensemble_size,
            )
            transmitters: Dict[
                int, Dict[str, ert.storage.StorageRecordTransmitter]
            ] = asyncio.get_event_loop().run_until_complete(future)
            assert len(transmitters) == ensemble_size
            futures = []
            for iens, transmitter in transmitters.items():
                # DO NOT export blob records as inputs
                if transmitter[record_name].record_type == ert.data.RecordType.BYTES:
                    continue
                futures.append(transmitter[record_name].load())
                if iens > 0 and iens % 50 == 0:
                    records = asyncio.get_event_loop().run_until_complete(
                        asyncio.gather(*futures)
                    )
                    for record in records:
                        inputs[record_name].append(record.data)
                    futures = []
            if len(futures) > 0:
                records = asyncio.get_event_loop().run_until_complete(
                    asyncio.gather(*futures)
                )
                for record in records:
                    inputs[record_name].append(record.data)

        elif record_source[0] == "resources":
            file_path = workspace / "resources" / record_source[1]
            collection = ert.data.load_collection_from_file(file_path)
            # DO NOT export blob records as inputs
            if collection.record_type == ert.data.RecordType.BYTES:
                continue
            assert collection.ensemble_size == ensemble_size
            for record in collection.records:
                inputs[record_name].append(record.data)
        else:
            raise ValueError(
                "Unknown record source location {}".format(record_source[0])
            )

    return inputs


def _prepare_export_responses(
    workspace: Path,
    experiment_name: str,
    ensemble: ert3.config.EnsembleConfig,
    ensemble_size: int,
) -> Dict[str, List[ert.data.record_data]]:
    outputs = defaultdict(list)
    responses = [elem.record for elem in ensemble.output]
    records_url = ert.storage.get_records_url(workspace, experiment_name)

    for record_name in responses:
        for iens in range(ensemble_size):
            url = f"{records_url}/{record_name}?realization_index={iens}"
            future = ert.storage.load_record(url, ert.data.RecordType.LIST_FLOAT)
            record = asyncio.get_event_loop().run_until_complete(future)
            outputs[record_name].append(record.data)
    return outputs


def export(
    workspace_root: Path,
    experiment_name: str,
    ensemble: ert3.config.EnsembleConfig,
    ensemble_size: int,
) -> None:

    experiment_root = (
        Path(workspace_root) / ert3.workspace.EXPERIMENTS_BASE / experiment_name
    )
    ert3.workspace.assert_experiment_exists(workspace_root, experiment_name)

    if not ert3.workspace.experiment_has_run(workspace_root, experiment_name):
        raise ValueError("Cannot export experiment that has not been carried out")

    parameters = _prepare_export_parameters(
        workspace_root, experiment_name, ensemble, ensemble_size
    )
    responses = _prepare_export_responses(
        workspace_root, experiment_name, ensemble, ensemble_size
    )
    data: List[Dict[str, Dict[str, Any]]] = []

    for iens in range(ensemble_size):
        inputs = {record: data[iens] for record, data in parameters.items()}
        outputs = {record: data[iens] for record, data in responses.items()}
        data.append({"input": inputs, "output": outputs})

    with open(experiment_root / "data.json", "w", encoding="utf-8") as f:
        json.dump(data, f)
