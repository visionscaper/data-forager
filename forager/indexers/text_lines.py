from typing import Optional, List, Tuple, Protocol
from dataclasses import dataclass

import os
import sys

from basics.base import Base
from tqdm import tqdm

from forager.index_stores.common import IndexStoreInterface


@dataclass
class SampleData:

    sample_bytes: bytes
    file_path: str

class SampleGeneratorInterface(Protocol):

    def prepare(self, text_file_path: str):
        """
        Prepare sample generation from a new input text file

        :param text_file_path: path to text file

        :return:
        """
        ...

    def create_samples(self, text_line: bytes) -> List[SampleData]:
        """
        Creates one or more samples from the given text_line and stores it in one or multiple different files.
        The path to the file(s) in which the samples are stores are also returned.

        IMPORTANT: it is assumed that each sample returned is stored in a file sequentially in the same order.
                   This must also hold over multiple function calls. This is important because the byte offset
                   of a sample is derived from the order the samples are returned.

        :param text_line: Text line in bytes from text_file_path, provided in the prepare phase.
                          The function needs to choose a text encoding itself

        :return: List of DataSample objects. For each created sample the following is given:
            * Its representation in bytes, as used to store the sample
            * The file path to where the sample is stored

        """
        ...

    def finish(self, is_last_file: bool):
        """
        Finish generation of samples from text lines of input file at the `text_file_path` given in the prepare() phase.

        is_last_file: indicates if the input text file was the last file to be processed

        :return:
        """
        ...


class NOOPSampleGenerator:

    def __init__(self):
        self._current_text_file = None

    def prepare(self, text_file_path: str):
        self._current_text_file = text_file_path

    def create_samples(self, text_line: bytes) -> List[SampleData]:
        return [SampleData(text_line, self._current_text_file)]

    def finish(self):
        self._current_text_file = None


def noop_sample_processing(text_line: bytes, text_file_path: str) -> List[SampleData]:

    return [SampleData(text_line, text_file_path)]


class FileTextLinesIndexer(Base):

    def __init__(
        self,
        input_file_paths: List[str],
        index_store: IndexStoreInterface,
        sample_generator: Optional[SampleGeneratorInterface] = None,
        description: Optional[str] = None,
        name: Optional[str] = None
    ):
        super().__init__(pybase_logger_name=name)

        if sample_generator is None:
            sample_generator = NOOPSampleGenerator()

        if description is None:
            description = "Indexing"

        self._input_file_paths = input_file_paths
        self._index_store = index_store

        self._sample_generator = sample_generator
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
            sys.stdout.write(
                f"{self._description}: \n"
                f"{input_file_path}\n"
            )
            sys.stdout.flush()

            self._sample_generator.prepare(input_file_path)

            with open(input_file_path, "rb") as f:
                file_size = os.fstat(f.fileno()).st_size
                pbar = tqdm(unit="bytes", total=file_size, file=sys.stdout)

                while text_line := f.readline():

                    num_text_line_bytes = len(text_line)

                    sample_data_list = self._sample_generator.create_samples(text_line)

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
                    sys.stdout.flush()

                pbar.close()

            is_last_file = input_file_path == self._input_file_paths[-1]
            self._sample_generator.finish(is_last_file=is_last_file)

            sys.stdout.write('\n\n')
            sys.stdout.flush()
