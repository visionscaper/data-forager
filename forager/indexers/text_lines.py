from typing import Optional, List, Tuple, Protocol
import os

from basics.base import Base
from tqdm import tqdm

from forager.index_stores.common import IndexStoreInterface


class ProcessSampleFunc(Protocol):

    def __call__(self, sample_text: str, text_file_path: str) -> Tuple[
        bytes,
        str
    ]:
        """
        Converts the sample_text in to sample data and appends it to a file. The name of the file is also returned.

        :param sample_text: Sample text to be converted.
        :param text_file_path: Source of text sample.

        :return: Tuple:
            sample data,
            file path where the sample data is stored
        """
        ...

def noop_sample_processing(sample_text: str, text_file_path: str) -> Tuple[
    bytes,
    str
]:
    return bytes(sample_text, 'utf-8'), text_file_path


class FileTextLinesIndexer(Base):

    def __init__(
        self,
        input_file_paths: List[str],
        index_store: IndexStoreInterface,
        process_sample_func: Optional[ProcessSampleFunc] = None,
        processing_description: Optional[str] = None,
        name: Optional[str] = None
    ):
        super().__init__(pybase_logger_name=name)

        if process_sample_func is None:
            process_sample_func = noop_sample_processing

        if processing_description is None:
            processing_description = "Indexing"
        else:
            processing_description = f"{processing_description} + Indexing"

        self._input_file_paths = input_file_paths
        self._index_store = index_store

        self._process_sample_func = process_sample_func
        self._processing_description = processing_description

    def __call__(self):
        self._index_store.init_store()

        byte_offset = 0
        for input_file_path in self._input_file_paths:
            self._log.info(f"{self._processing_description}: \n"
                           f"{input_file_path}")

            with open(input_file_path, "r") as f:
                file_size = os.fstat(f.fileno()).st_size
                pbar = tqdm(desc=self._processing_description, unit="bytes", total=file_size)

                while text_line := f.readline():

                    sample_data, sample_file_path = self._process_sample_func(
                        text_line,
                        input_file_path,
                    )
                    num_bytes = len(sample_data)
                    self._index_store.add_sample(
                        file_location=sample_file_path,
                        byte_offset=byte_offset,
                        num_bytes=num_bytes
                    )

                    byte_offset += num_bytes
                    pbar.update(num_bytes)
