import os.path
import shutil
from typing import Optional, TextIO, BinaryIO

import numpy as np

from basics.base import Base

from data_forager.index_stores.common import IndexStoreInterface
from data_forager.sample_index import SampleIndex


class IndexStore(Base, IndexStoreInterface):

    def __init__(self, base_path: str, index_data_folder: str = "index", name: Optional[str] = None):
        """

        :param base_path: Base path where the index files are stored.
            File paths in file_location.txt are stored relative to this path.

        :param name: Name of instance, if not provided, the classname will be used
        """

        super().__init__(pybase_logger_name=name)

        self._base_path = os.path.abspath(base_path)
        self._index_data_path = os.path.join(self._base_path, index_data_folder)
        self._file_locations = []

        # File handles for buffered writing
        self._file_location_handle: Optional[TextIO] = None
        self._sample_locations_handle: Optional[BinaryIO] = None

    def init_store(self):
        if os.path.exists(self._index_data_path):
            raise ValueError(f"Provided index path already exists: {self._index_data_path}")

        os.makedirs(self._index_data_path, exist_ok=True)

        # Open file handles for writing
        self._file_location_handle = open(
            os.path.join(self._index_data_path, "file_location.txt"), "a"
        )
        self._sample_locations_handle = open(
            os.path.join(self._index_data_path, "sample_locations.bin"), "ab"
        )

    def add_sample(self, file_location: str, byte_offset: int, num_bytes: int):
        """
        :param file_location: Absolute or relative path to the sample file.
            Will be stored as a path relative to base_path.
        :param byte_offset:
        :param num_bytes:
        :return:
        """
        if file_location not in self._file_locations:
            self._file_locations.append(file_location)
            # Store as relative path for portability
            relative_path = os.path.relpath(file_location, self._base_path)
            self._file_location_handle.write(relative_path + '\n')

        file_index = self._file_locations.index(file_location)

        sample_location_bytes = np.array([file_index, byte_offset, num_bytes], dtype=np.uint64).tobytes()
        self._sample_locations_handle.write(sample_location_bytes)

    def close(self):
        """Close file handles and flush buffered data."""
        if self._file_location_handle is not None:
            self._file_location_handle.close()
            self._file_location_handle = None

        if self._sample_locations_handle is not None:
            self._sample_locations_handle.close()
            self._sample_locations_handle = None

    def __del__(self):
        self.close()

    def load(self) -> SampleIndex:
        with open(os.path.join(self._index_data_path, "file_location.txt"), "r") as f:
            relative_locations = [loc[:-1] if loc[-1]=='\n' else loc for loc in f.readlines()]

        # Resolve relative paths against base_path
        file_locations = [
            os.path.join(self._base_path, loc) for loc in relative_locations
        ]

        with open(os.path.join(self._index_data_path, "sample_locations.bin"), "rb") as f:
            data = f.read()
            sample_locations = np.frombuffer(data, dtype=np.uint64)
            sample_locations = sample_locations.reshape((-1, 3))

        return SampleIndex(file_locations, sample_locations)

    def exists(self) -> bool:
        """Check if the index already exists."""
        sample_locations = os.path.join(self._index_data_path, "sample_locations.bin")
        file_location = os.path.join(self._index_data_path, "file_location.txt")
        return os.path.exists(sample_locations) and os.path.exists(file_location)

    def clear(self) -> None:
        """Remove the index directory and all its contents."""
        if os.path.exists(self._index_data_path):
            shutil.rmtree(self._index_data_path)
            self._file_locations = []
