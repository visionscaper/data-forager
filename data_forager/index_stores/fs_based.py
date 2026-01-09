import os.path
from typing import Optional

import numpy as np

from basics.base import Base

from data_forager.sample_index import SampleIndex


class IndexStore(Base):

    def __init__(self, base_path: str, index_data_folder: str = "index", name: Optional[str] = None):
        """

        :param base_path: Base path where the index files are stored.

        :param name: Name of instance, if not provided, the classname will be used
        """

        super().__init__(pybase_logger_name=name)

        self._index_data_path = os.path.join(base_path, index_data_folder)
        self._file_locations = []

    def init_store(self):
        if os.path.exists(self._index_data_path):
            raise ValueError(f"Provided index path already exists: {self._index_data_path}")

        os.makedirs(self._index_data_path, exist_ok=True)

    def add_sample(self, file_location: str, byte_offset: int, num_bytes: int):
        """
        :param file_location:
        :param byte_offset:
        :param num_bytes:
        :return:
        """
        if file_location not in self._file_locations:
            self._file_locations.append(file_location)
            with open(os.path.join(self._index_data_path, "file_location.txt"), "a") as f:
                f.writelines([file_location+'\n'])

        file_index = self._file_locations.index(file_location)

        with open(os.path.join(self._index_data_path, "sample_locations.bin"), "ab") as f:
            sample_location_bytes = np.array([file_index, byte_offset, num_bytes], dtype=np.uint64).tobytes()

            f.write(sample_location_bytes)

    def load(self) -> SampleIndex:
        with open(os.path.join(self._index_data_path, "file_location.txt"), "r") as f:
            file_locations = [loc[:-1] if loc[-1]=='\n' else loc for loc in f.readlines()]

        with open(os.path.join(self._index_data_path, "sample_locations.bin"), "rb") as f:
            data = f.read()
            sample_locations = np.frombuffer(data, dtype=np.uint64)
            sample_locations = sample_locations.reshape((-1, 3))

        return SampleIndex(file_locations, sample_locations)
