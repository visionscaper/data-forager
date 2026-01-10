from typing import Protocol

from data_forager.sample_index import SampleIndex


class IndexStoreInterface(Protocol):

    def init_store(self):
        """Initialize the store for writing. Must be called before add_sample()."""
        ...

    def add_sample(self, file_location: str, byte_offset: int, num_bytes: int):
        """Add a sample location to the index."""
        ...

    def close(self):
        """Close the store, flushing any buffered data. Must be called after all samples are added."""
        ...

    def load(self) -> SampleIndex:
        """Load the index from the store."""
        ...

    def exists(self) -> bool:
        """Check if the index already exists."""
        ...

    def clear(self) -> None:
        """Remove the index."""
        ...
