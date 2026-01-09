from typing import Protocol

from data_forager.sample_index import SampleIndex


class IndexStoreInterface(Protocol):

    def init_store(self):
        pass

    def add_sample(self, file_location: str, byte_offset: int, num_bytes: int):
        pass

    def load(self) -> SampleIndex:
        pass
