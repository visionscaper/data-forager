from dataclasses import dataclass
from typing import Optional, List, Tuple, Protocol
import os

from basics.base import Base
from tqdm import tqdm

from forager.index_stores.common import IndexStoreInterface


@dataclass
class SampleData:

    sample_bytes: bytes
    file_path: str


class CreateSamplesFunc(Protocol):

    def __call__(self, text_line: bytes, text_file_path: str) -> List[SampleData]:
        """
        Creates one or more samples from the given text_line and stores it in one or multiple different files.
        The path to the file(s) in which the samples are stores are also returned.

        IMPORTANT: it is assumed that each sample returned is stored in a file sequentially in the same order.
                   This must also hold over multiple function calls. This is important because the byte offset
                   of a sample is derived from the order the samples are returned.

        :param text_line: Text line in bytes, the function needs to choose a text encoding itself
        :param text_file_path: Source of text line.

        :return: List of DataSample objects. For each created sample the following is given:
            * Its representation in bytes, as used to store the sample
            * The file path to where the sample is stored

        """
        ...


def noop_sample_processing(text_line: bytes, text_file_path: str) -> List[SampleData]:
    return [SampleData(text_line, text_file_path)]


class FileTextLinesIndexer(Base):

    def __init__(
        self,
        input_file_paths: List[str],
        index_store: IndexStoreInterface,
        create_samples_func: Optional[CreateSamplesFunc] = None,
        description: Optional[str] = None,
        name: Optional[str] = None
    ):
        super().__init__(pybase_logger_name=name)

        if create_samples_func is None:
            create_samples_func = noop_sample_processing

        if description is None:
            description = "Indexing"

        self._input_file_paths = input_file_paths
        self._index_store = index_store

        self._create_samples_func = create_samples_func
        self._description = description

    def __call__(self):
        """
        IMPORTANT: input files are always read in binary mode; applying a text encoding is up to the user.
                   E.g. through process_sample_func and/or when processing the data Dataset::_process_sample()

        :return:
        """
        self._index_store.init_store()

        byte_offset_map = {}

        for input_file_path in self._input_file_paths:
            self._log.info(
                f"{self._description}: \n"
                f"{input_file_path}"
            )

            with open(input_file_path, "rb") as f:
                file_size = os.fstat(f.fileno()).st_size
                pbar = tqdm(desc=self._description, unit="bytes", total=file_size)

                while text_line := f.readline():

                    num_text_line_bytes = len(text_line)

                    sample_data_list = self._create_samples_func(
                        text_line,
                        input_file_path,
                    )

                    for sample_data in sample_data_list:
                        file_location = sample_data.file_path
                        byte_offset = byte_offset_map.get(file_location, 0)
                        num_bytes = len(sample_data.sample_bytes)

                        self._index_store.add_sample(
                            file_location=file_location,
                            byte_offset=byte_offset,
                            num_bytes=num_bytes
                        )

                        byte_offset_map[file_location] = byte_offset + num_bytes

                    pbar.update(num_text_line_bytes)
