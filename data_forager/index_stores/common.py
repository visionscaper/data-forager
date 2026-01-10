from typing import Protocol

from data_forager.sample_index import SampleIndex


class IndexStoreInterface(Protocol):

    def init_store(self):
        ...

    def add_sample(self, file_location: str, byte_offset: int, num_bytes: int):
        ...

    def load(self) -> SampleIndex:
        ...

    def exists(self) -> bool:
        ...

    def clear(self) -> None:
        ...
