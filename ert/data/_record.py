import json
import pathlib
import shutil
import uuid
from abc import abstractmethod
from enum import Enum, auto
from pathlib import Path
from itertools import repeat

from typing import (
    Any,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Union,
    Dict,
    Iterator,
)
import aiofiles

# Type hinting for wrap must be turned off until (1) is resolved.
# (1) https://github.com/Tinche/aiofiles/issues/8
from aiofiles.os import wrap  # type: ignore
from pydantic import (
    BaseModel,
    StrictBytes,
    StrictFloat,
    StrictInt,
    StrictStr,
    NonNegativeInt,
    validator,
    root_validator,
)

_copy = wrap(shutil.copy)

strict_number = Union[StrictInt, StrictFloat]
numerical_record_data = Union[
    List[strict_number],
    Dict[StrictStr, strict_number],
    Dict[StrictInt, strict_number],
]
blob_record_data = StrictBytes
record_data = Union[numerical_record_data, blob_record_data]


def parse_json_key_as_int(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {int(k): v for k, v in obj.items()}
    return obj


class _DataElement(BaseModel):
    class Config:
        validate_all = True
        validate_assignment = True
        extra = "forbid"
        allow_mutation = False
        arbitrary_types_allowed = True


RecordIndex = Tuple[Union[StrictInt, StrictStr], ...]


def _build_record_index(
    data: numerical_record_data,
) -> RecordIndex:
    if isinstance(data, MutableMapping):
        return tuple(data.keys())
    else:
        return tuple(range(len(data)))


class RecordType(str, Enum):
    LIST_FLOAT = "LIST_FLOAT"
    MAPPING_INT_FLOAT = "MAPPING_INT_FLOAT"
    MAPPING_STR_FLOAT = "MAPPING_STR_FLOAT"
    BYTES = "BYTES"


class Record(_DataElement):
    data: record_data
    record_type: Optional[RecordType] = None

    @validator("record_type", pre=True)
    def record_type_validator(
        cls,
        record_type: Optional[RecordType],
        values: Dict[str, Any],
    ) -> Optional[RecordType]:
        if record_type is None and "data" in values:
            data = values["data"]
            if isinstance(data, list):
                if not data or isinstance(data[0], (int, float)):
                    return RecordType.LIST_FLOAT
            elif isinstance(data, bytes):
                return RecordType.BYTES
            elif isinstance(data, Mapping):
                if not data:
                    return RecordType.MAPPING_STR_FLOAT
                if isinstance(list(data.keys())[0], (int, float)):
                    return RecordType.MAPPING_INT_FLOAT
                if isinstance(list(data.keys())[0], str):
                    return RecordType.MAPPING_STR_FLOAT
        return record_type

    def get_instance(self) -> "Record":
        if self.record_type is not None:
            if self.record_type == RecordType.BYTES:
                return BlobRecord(data=self.data)
        return NumericalRecord(data=self.data)


class NumericalRecord(Record):
    data: numerical_record_data
    index: Optional[RecordIndex] = None

    @validator("index", pre=True)
    def index_validator(
        cls,
        index: Optional[RecordIndex],
        values: Dict[str, Any],
    ) -> Optional[RecordIndex]:
        if index is None and "data" in values:
            index = _build_record_index(values["data"])
        return index

    @root_validator(skip_on_failure=True)
    def ensure_consistent_index(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        assert (
            "data" in record and "index" in record
        ), "both data and index must be defined for a record"
        norm_record_index = _build_record_index(record["data"])
        assert (
            norm_record_index == record["index"]
        ), f"inconsistent index {norm_record_index} vs {record['index']}"
        return record


class BlobRecord(Record):
    data: blob_record_data


class RecordCollection(_DataElement):
    records: Union[Tuple[NumericalRecord, ...], Tuple[BlobRecord, ...]]
    ensemble_size: Optional[NonNegativeInt] = None

    @property
    def record_type(self) -> Optional[RecordType]:
        return self.records[0].record_type

    @validator("records")
    def records_validator(
        cls, records: Union[Tuple[NumericalRecord, ...], Tuple[BlobRecord, ...]]
    ) -> Union[Tuple[NumericalRecord, ...], Tuple[BlobRecord, ...]]:
        assert len(records) > 0
        record_type = records[0].record_type
        for record in records[1:]:
            if record.record_type != record_type:
                raise ValueError("Ensemble records must have an invariant record type")
        return records

    @validator("ensemble_size", pre=True, always=True)
    def ensemble_size_validator(
        cls, ensemble_size: Optional[NonNegativeInt], values: Dict[str, Any]
    ) -> Optional[NonNegativeInt]:
        if "records" in values:
            if ensemble_size is None:
                ensemble_size = len(values["records"])
            elif len(values["records"]) > 1 and ensemble_size != len(values["records"]):
                raise ValueError(
                    "ensemble_size incompatible with the number of records"
                )
        return ensemble_size

    def __iter__(self) -> Iterator[Record]:  # type: ignore
        if self.ensemble_size != len(self.records):
            assert self.ensemble_size is not None  # making mypy happy
            return repeat(self.records[0], self.ensemble_size)
        return iter(self.records)

    def __getitem__(self, idx: int) -> Union[NumericalRecord, BlobRecord]:
        if self.ensemble_size != len(self.records):
            assert self.ensemble_size is not None  # making mypy happy
            if idx < 0 or idx >= self.ensemble_size:
                raise IndexError("record collection index out of range")
            return self.records[0]
        return self.records[idx]


class RecordCollectionMap(_DataElement):
    record_collections: Mapping[str, RecordCollection]
    record_names: Optional[Tuple[str, ...]] = None
    ensemble_size: Optional[int] = None

    @validator("record_names", pre=True, always=True)
    def record_names_validator(
        cls, record_names: Optional[Tuple[str, ...]], values: Dict[str, Any]
    ) -> Optional[Tuple[str, ...]]:
        if record_names is None and "record_collections" in values:
            record_collections = values["record_collections"]
            record_names = tuple(record_collections.keys())
        return record_names

    @validator("ensemble_size", pre=True, always=True)
    def ensemble_size_validator(
        cls, ensemble_size: Optional[int], values: Dict[str, Any]
    ) -> Optional[int]:
        if (
            ensemble_size is None
            and "record_collections" in values
            and "record_names" in values
        ):
            record_names = values["record_names"]
            assert len(record_names) > 0
            record_collections = values["record_collections"]
            assert len(record_collections) > 0
            first_record = record_collections[record_names[0]]
            try:
                ensemble_size = first_record.ensemble_size
            except AttributeError:
                ensemble_size = first_record["records"].ensemble_size
        assert ensemble_size is not None and ensemble_size > 0
        return ensemble_size

    @root_validator(skip_on_failure=True)
    def ensure_consistent_ensemble_size(
        cls, record_collection_map: Dict[str, Any]
    ) -> Dict[str, Any]:
        ensemble_size = record_collection_map["ensemble_size"]
        for collection in record_collection_map["record_collections"].values():
            if ensemble_size != collection.ensemble_size:
                raise AssertionError("Inconsistent ensemble record size")
        return record_collection_map

    @root_validator(skip_on_failure=True)
    def ensure_consistent_record_names(
        cls, record_collection_map: Dict[str, Any]
    ) -> Dict[str, Any]:
        assert "record_names" in record_collection_map
        record_names = tuple(record_collection_map["record_collections"].keys())
        assert record_collection_map["record_names"] == record_names
        return record_collection_map

    def __len__(self) -> int:
        assert self.record_names is not None
        return len(self.record_names)


class RecordTransmitterState(Enum):
    transmitted = auto()
    not_transmitted = auto()


class RecordTransmitterType(Enum):
    in_memory = auto()
    ert_storage = auto()
    shared_disk = auto()


class RecordTransmitter:
    def __init__(self, transmitter_type: RecordTransmitterType) -> None:
        self._state = RecordTransmitterState.not_transmitted
        self._uri: str = ""
        self._record_type: Optional[RecordType] = None
        self._transmitter_type: RecordTransmitterType = transmitter_type

    def _set_transmitted_state(
        self, uri: str, record_type: Optional[RecordType]
    ) -> None:
        self._state = RecordTransmitterState.transmitted
        self._uri = uri
        self._record_type = record_type

    def is_transmitted(self) -> bool:
        return self._state == RecordTransmitterState.transmitted

    @property
    def transmitter_type(self) -> RecordTransmitterType:
        return self._transmitter_type

    @abstractmethod
    async def _load_numerical_record(self) -> NumericalRecord:
        pass

    @abstractmethod
    async def _load_blob_record(self) -> BlobRecord:
        pass

    async def load(self) -> Record:
        if not self.is_transmitted():
            raise RuntimeError("cannot load untransmitted record")
        if self._record_type != RecordType.BYTES:
            return await self._load_numerical_record()
        return await self._load_blob_record()

    @abstractmethod
    async def _transmit_numerical_record(self, record: NumericalRecord) -> str:
        pass

    @abstractmethod
    async def _transmit_blob_record(self, record: BlobRecord) -> str:
        pass

    async def transmit_record(self, record: Record) -> None:
        if self.is_transmitted():
            raise RuntimeError("Record already transmitted")
        if isinstance(record, NumericalRecord):
            uri = await self._transmit_numerical_record(record)
        elif isinstance(record, BlobRecord):
            uri = await self._transmit_blob_record(record)
        else:
            raise TypeError(f"Record type not supported {type(record)}")
        self._set_transmitted_state(uri, record_type=record.record_type)

    async def transmit_data(
        self,
        data: record_data,
    ) -> None:
        if self.is_transmitted():
            raise RuntimeError("Record already transmitted")
        record = Record(data=data)
        await self.transmit_record(record.get_instance())

    async def transmit_file(
        self,
        file: Path,
        mime: str,
    ) -> None:
        if self.is_transmitted():
            raise RuntimeError("Record already transmitted")
        if mime == "application/json":
            async with aiofiles.open(str(file), mode="r") as f:
                contents = await f.read()
                num_record = NumericalRecord(data=json.loads(contents))
            uri = await self._transmit_numerical_record(num_record)
            self._set_transmitted_state(uri, num_record.record_type)
        elif mime == "application/octet-stream":
            async with aiofiles.open(str(file), mode="rb") as f:  # type: ignore
                contents = await f.read()
                blob_record = BlobRecord(data=contents)
            uri = await self._transmit_blob_record(blob_record)
            self._set_transmitted_state(uri, blob_record.record_type)
        else:
            raise NotImplementedError(
                "cannot transmit file unless mime is application/json"
                f" or application/octet-stream, was {mime}"
            )

    async def dump(self, location: Path) -> None:
        if not self.is_transmitted():
            raise RuntimeError("cannot dump untransmitted record")
        record = await self.load()
        if isinstance(record, NumericalRecord):
            async with aiofiles.open(str(location), mode="w") as f:
                await f.write(json.dumps(record.data))
        else:
            async with aiofiles.open(str(location), mode="wb") as f:  # type: ignore
                await f.write(record.data)  # type: ignore


class SharedDiskRecordTransmitter(RecordTransmitter):
    def __init__(self, name: str, storage_path: Path):
        super().__init__(RecordTransmitterType.shared_disk)
        self._storage_path = storage_path
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._concrete_key = f"{name}_{uuid.uuid4()}"
        self._storage_uri = self._storage_path / self._concrete_key

    async def _transmit_numerical_record(self, record: NumericalRecord) -> str:
        contents = json.dumps(record.data)
        async with aiofiles.open(self._storage_uri, mode="w") as f:
            await f.write(contents)
        return str(self._storage_uri)

    async def _transmit_blob_record(self, record: BlobRecord) -> str:
        async with aiofiles.open(self._storage_uri, mode="wb") as f:
            await f.write(record.data)
        return str(self._storage_uri)

    async def _load_numerical_record(self) -> NumericalRecord:
        async with aiofiles.open(str(self._uri)) as f:
            contents = await f.read()
        if self._record_type == RecordType.MAPPING_INT_FLOAT:
            data = json.loads(contents, object_hook=parse_json_key_as_int)
        else:
            data = json.loads(contents)
        return NumericalRecord(data=data)

    async def _load_blob_record(self) -> BlobRecord:
        async with aiofiles.open(str(self._uri), mode="rb") as f:
            data = await f.read()
        return BlobRecord(data=data)

    async def dump(self, location: Path) -> None:
        if not self.is_transmitted():
            raise RuntimeError("cannot dump untransmitted record")
        await _copy(self._uri, str(location))


class InMemoryRecordTransmitter(RecordTransmitter):
    def __init__(self, name: str):
        super().__init__(RecordTransmitterType.in_memory)
        self._name = name
        self._record: Record

    async def _transmit_numerical_record(self, record: NumericalRecord) -> str:
        self._record = record
        return "in_memory"

    async def _transmit_blob_record(self, record: BlobRecord) -> str:
        self._record = record
        return "in_memory"

    async def _load_numerical_record(self) -> NumericalRecord:
        return NumericalRecord(data=self._record.data)

    async def _load_blob_record(self) -> BlobRecord:
        return BlobRecord(data=self._record.data)


def load_collection_from_file(
    file_path: pathlib.Path, blob_record: bool = False, ens_size: int = 1
) -> RecordCollection:
    if blob_record:
        with open(file_path, "rb") as fb:
            return RecordCollection(
                records=[BlobRecord(data=fb.read())],
                ensemble_size=ens_size,
            )

    with open(file_path, "r") as f:
        raw_ensrecord = json.load(f)
    return RecordCollection(
        records=[NumericalRecord(data=raw_record) for raw_record in raw_ensrecord]
    )
