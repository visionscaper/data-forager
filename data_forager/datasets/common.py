import abc
from typing import Optional, Union, Dict, List, Protocol, Any

import numpy as np

from basics.base import Base

from data_forager.sample_index import SampleIndex, SampleLocation
from data_forager.index_stores.fs_based import IndexStore as FSBasedIndexStore


class Dataset(Base, metaclass=abc.ABCMeta):

    @classmethod
    def create_from_index_on_filesystem(
        cls,
        base_path: str,
        index_data_folder: str = "index",
        name: Optional[str] = None,
        **kwargs
    ):
        """

        USAGE:
        training_set = create_from_index_on_filesystem('.')
        training_set.initialize()

        random_indices = list(range(len(training_set)))
        random.shuffle(random_indices)

        for idx in random_indices:
            sample = training_set[idx]
            print(sample)

        :param base_path:
        :param index_data_folder:
        :param name:
        :param kwargs: other keyword arguments that will be passed to the constructor of the Dataset

        :return:
        """
        index_store = FSBasedIndexStore(
            base_path,
            index_data_folder=index_data_folder,
        )

        sample_index = index_store.load()

        return cls(sample_index, name=name, **kwargs)

    def __init__(self, sample_index: SampleIndex, name: Optional[str] = None, **kwargs):
        """
        See create_from_index_on_filesystem for usage example.

        :param sample_index:
        :param name:
        """
        super().__init__(pybase_logger_name=name, **kwargs)

        self._sample_index = sample_index
        self._file_handles = None
        self._is_initialized = False

    def initialize(self):
        self._open_files()
        self._is_initialized = True

    @property
    def is_initialized(self):
        return self._is_initialized

    def __getitem__(self, idx: Union[int, slice]) -> Union[Dict, List[Dict]]:
        if not self.is_initialized:
            self.initialize()

        if type(idx) is slice:
            indices = range(idx.start, idx.stop, idx.step)
            return [self._get_sample(self._sample_index[i]) for i in indices]
        elif type(idx) is int:
            return self._get_sample(self._sample_index[idx])
        else:
            raise ValueError(f"idx must be int or slice")

    @abc.abstractmethod
    def _process_sample(self, sample_bytes: bytes) -> Any:
        raise NotImplementedError("Implement this method in your subclass")

    def _open_files(self):
        self._log.debug(f"Opening sample files for {self} ...")
        self._file_handles = {
            file_location: open(file_location, mode="rb")
            for file_location in self._sample_index.file_locations
        }

    def _close_files(self):
        if not self.is_initialized:
            return

        self._log.debug(f"Closing sample files for {self} ...")
        for file_location, file_handle in self._file_handles.items():
            try:
                file_handle.close()
            except Exception as e:
                self._log.error(
                    f"Unable to close file: \n"
                    f"{file_location}\n"
                    f"{e}")

    def _get_sample_bytes(self, location: SampleLocation) -> bytes:
        try:
            file_handle = self._file_handles[location.file_path]
        except IndexError:
            raise IndexError(f"Unable to get sample from unknown file:\n"
                             f"Sample Location:\n"
                             f"{location}")

        try:
            file_handle.seek(location.byte_offset)
            return file_handle.read(location.num_bytes)
        except Exception as e:
            raise IndexError(f"Unable to read sample bytes:\n"
                             f"Sample Location:\n"
                             f"{location}") from e

    def _get_sample(self, location: SampleLocation):
        return self._process_sample(self._get_sample_bytes(location))

    def __len__(self):
        return len(self._sample_index)

    def __del__(self):
        self._close_files()


class SubsampledDataset:
    """
    Wrapper that provides a subsampled view of a dataset.

    Randomly selects a subset of indices from the wrapped dataset, allowing
    for faster iteration through epochs when testing or debugging.

    :param dataset: The dataset to wrap (must support __len__ and __getitem__).
    :param subsample_factor: Fraction of the dataset to use (must be between 0 and 1).
    :param seed: Random seed for reproducibility. If None, sampling is random.
    :param random_order: If False (default), indices are sorted for better disk
        read locality. If True, indices are kept in random order, which can be
        used as a randomizer.
    """

    def __init__(
        self,
        dataset,
        subsample_factor: float,
        seed: int | None = None,
        random_order: bool = False,
    ):
        if not 0 < subsample_factor <= 1:
            raise ValueError(
                f"subsample_factor must be between 0 (exclusive) and 1 (inclusive), "
                f"got {subsample_factor}"
            )

        self._dataset = dataset
        self._subsample_factor = subsample_factor

        n_full = len(dataset)
        n_sub = int(subsample_factor * n_full)

        # Sample indices without replacement
        rng = np.random.default_rng(seed)
        indices = rng.choice(n_full, size=n_sub, replace=False)

        # Sort for cache locality unless random order is requested
        if not random_order:
            indices.sort()

        # Convert to Python list of ints for underlying dataset compatibility
        self._indices: list[int] = indices.tolist()

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._dataset[self._indices[idx]]
