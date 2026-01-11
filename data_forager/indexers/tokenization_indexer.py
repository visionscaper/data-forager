from typing import Callable, List, Optional

import logging
import os
from pathlib import Path

import json

import numpy as np

from basics.base import Base
from basics.logging import get_logger

module_logger = get_logger(os.path.basename(__file__))

from data_forager.index_stores.common import IndexStoreInterface
from data_forager.index_stores.fs_based import IndexStore as FSBasedIndexStore
from data_forager.indexers.text_lines import SampleData, FileTextLinesIndexer, SampleGeneratorInterface
from data_forager.utils import find_files_recursive, natural_sort

TokenizerFunc = Callable[[str], List[int]]
ProcessTextLineFunc = Callable[[bytes], str]


def get_text_from_jsonl(jsonl_bytes: bytes, text_key: str = "text", text_encoding: str = "utf-8") -> str:
    jsonl_text = jsonl_bytes.decode(text_encoding)
    data = json.loads(jsonl_text)
    return data[text_key]


def create_tokenize_and_index_jsonl_text_func(
    tokenizer_func: TokenizerFunc,
    eos_idx: int,
    input_base_path: Optional[str] = None,
    input_file_paths: Optional[List[str]] = None,
    output_base_path: Optional[str] = None,
    index_store: Optional[IndexStoreInterface] = None,
    process_text_line_func: Optional[ProcessTextLineFunc] = None,
    logger: Optional[logging.Logger] = None,
    name: Optional[str] = None,
    **sample_generator_kwargs,
) -> FileTextLinesIndexer:
    """
    Create a pipeline to tokenize text from JSONL files and create an index for random access.

    The pipeline:
     * Tokenizes text from input JSONL objects
     * Stores the token data in bin files under "tokenized-samples" folder
     * Stores index data under "index" folder

    Usage:
    ```python
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    def tokenize_text(text: str) -> List[int]:
        return enc.encode_ordinary(text)

    # Option 1: Scan directory for JSONL files, output to same directory
    indexer = create_tokenize_and_index_jsonl_text_func(
        tokenizer_func=tokenize_text,
        eos_idx=enc.eot_token,
        input_base_path='./data',
        sample_size=1024,
    )

    # Option 2: Explicit input files and output path
    indexer = create_tokenize_and_index_jsonl_text_func(
        tokenizer_func=tokenize_text,
        eos_idx=enc.eot_token,
        input_file_paths=['./data/train.jsonl'],
        output_base_path='./output',
        sample_size=1024,
    )

    # Run tokenization and indexing
    indexer()
    ```

    :param tokenizer_func: Function used to tokenize text.
    :param eos_idx: EOS token index, known by the used Tokenizer.
    :param input_base_path: Path to directory containing JSONL files (searched recursively).
        Used as fallback for output if `output_base_path` is not provided.
    :param input_file_paths: List of file paths to process. If provided, these are used
        instead of scanning `input_base_path` for JSONL files.
    :param output_base_path: Base path for output (index and tokenized samples).
        If not provided, `input_base_path` is used.
    :param index_store: Index store to use. If provided, this is used instead of
        creating a new FSBasedIndexStore.
    :param process_text_line_func: Function used to process text lines.
        By default, this converts input JSON lines to dicts and returns the "text" field.
        See function get_text_from_jsonl().
    :param logger: Logger to use. If not provided, uses module logger.
    :param name: Name of the indexer, used for logging purposes.
    :param sample_generator_kwargs: Other kwargs passed to TokenizedSampleGenerator
        (e.g., sample_size, token_dtype, base_output_path).

    :raises ValueError: If both `input_base_path` and `input_file_paths` are None.
    :raises ValueError: If `index_store` is None and both `output_base_path` and
        `input_base_path` are None.

    :return: FileTextLinesIndexer instance that can be called to run tokenization
        and indexing.
    """
    if logger is None:
        logger = module_logger

    # Validate input source
    if input_base_path is None and input_file_paths is None:
        raise ValueError(
            "Either input_base_path or input_file_paths must be provided"
        )

    # Determine output base path
    effective_output_base_path = output_base_path or input_base_path

    # Validate output destination
    if index_store is None and effective_output_base_path is None:
        raise ValueError(
            "Either index_store, output_base_path, or input_base_path must be provided "
            "to determine where to store the index"
        )

    logger.info(f"Output base path: {effective_output_base_path}")

    if process_text_line_func is None:
        process_text_line_func = get_text_from_jsonl

    if index_store is None:
        index_store = FSBasedIndexStore(
            base_path=effective_output_base_path,
        )

    if input_file_paths is None:
        logger.info(f"Scanning for JSONL files in: {input_base_path}")
        input_file_paths = find_files_recursive(
            input_base_path,
            extension_patterns=['*.jsonl', '*.JSONL']
        )
        # Assuming numbered files
        input_file_paths = natural_sort(input_file_paths)
        logger.info(f"Found {len(input_file_paths)} JSONL file(s)")

    # Set default base_output_path for tokenized samples if not provided in kwargs
    if 'base_output_path' not in sample_generator_kwargs:
        default_base_output_path = os.path.join(
            effective_output_base_path, "tokenized-samples"
        )
        logger.info(f"Tokenized samples output path: {default_base_output_path}")
        sample_generator_kwargs['base_output_path'] = default_base_output_path

    sample_generator = TokenizedSampleGenerator(
        process_text_line_func=process_text_line_func,
        tokenizer_func=tokenizer_func,
        eos_idx=eos_idx,
        **sample_generator_kwargs
    )

    return FileTextLinesIndexer(
        input_file_paths=input_file_paths,
        index_store=index_store,
        sample_generator=sample_generator,
        description="Tokenizing and indexing",
        name=name,
    )


class TokenizedSampleGenerator(Base, SampleGeneratorInterface):

    def __init__(
        self,
        process_text_line_func: ProcessTextLineFunc,
        tokenizer_func: TokenizerFunc,
        eos_idx: int,
        token_dtype: np.dtype = np.uint16,
        sample_size: Optional[int] = None,
        base_output_path: str = None,
        file_name_postfix: str = "tokenized-samples",
        name: Optional[str] = None
    ):
        """
        Tokenizes and indexed text into fixed length (`sample_size` not None) samples or
        samples variable of variable size, depending on the text (`sample_size` is not None).

        This callable performs the following steps:
        ## prepare ##
         * In the preparation step, create file to store tokenized samples, based on the input `text_file_path` and
           the given `base_output_path` and `file_name_postfix`
         * If the `base_output_path` is not given the `text_file_path` will be used + "/tokenized-samples"
        ## create_samples ##
         * To create tokenized text samples, processes incoming text line using `process_text_line_func`,
           e.g. convert JSONL in to dict and retrieve the sample text from it.
         * The resulting text is tokenized using `tokenizer_func`.
         * If a `sample_size` is given:
           The tokenized text is split into samples of length `sample_size` and stored in the
           file opened in the prepare step. Here `token_dtype` is used.
            - Trailing tokens will be combined with samples of a next text line
            - Tokens from different text samples will be separated by `eos_idx`
         * If a `sample_size` is not given:
           The tokenized text is immediately stored as is, in the file opened in the prepare step.
           Here `token_dtype` is used.
        ## finish ##
         * After all text lines are processed, the file holding the tokenized text samples is closed.
           When `sample_size` not None: Any final trailing tokens will be discarded, but only when the last
           input text file was processed.

        :param tokenizer_func:
        :param name:
        """
        super().__init__(pybase_logger_name=name)

        if sample_size is None:
            self._log.info(f"Tokenized text will NOT be broken in to samples of fixed length.")

        self._process_text_line_func = process_text_line_func
        self._tokenizer_func = tokenizer_func
        self._eos_idx = eos_idx
        self._token_dtype = token_dtype
        self._sample_size = sample_size
        self._base_output_path = base_output_path
        self._file_name_postfix = file_name_postfix

        self._current_samples_path = None
        self._current_samples_file = None

        self._rest_tokens = None

    def prepare(self, text_file_path: str):
        """
        ## prepare ##
         * In the preparation step, create file to store tokenized samples, based on the input `text_file_path` and
           the given `base_output_path` and `file_name_postfix`
         * If the `base_output_path` is not given the `text_file_path` will be used + "/tokenized-samples"

        :param text_file_path: path to text file

        :return:
        """
        input_file_path = os.path.dirname(text_file_path)
        input_file_name = Path(text_file_path).stem
        output_file_name = f"{input_file_name}-{self._file_name_postfix}.bin"

        output_path = self._base_output_path
        if self._base_output_path is None:
            output_path = os.path.join(input_file_path, "tokenized-samples")

        os.makedirs(output_path, exist_ok=True)

        output_file_path = os.path.join(output_path, output_file_name)
        if os.path.exists(output_file_path):
            raise FileExistsError(f"Tokenized samples file already exists: \n{output_file_path}")

        self._current_samples_path = output_file_path
        self._current_samples_file = open(output_file_path, "wb")

        self._log.debug(f"Tokenized samples file opened: \n"
                        f"{output_file_path}")

    def create_samples(self, text_line: bytes) -> List[SampleData]:
        """

        ## create_samples ##
         * To create tokenized text samples, processes incoming text line using `process_text_line_func`,
           e.g. convert JSONL in to dict and retrieve the sample text from it.
         * The resulting text is tokenized using `tokenizer_func`.
         * If a `sample_size` is given:
           The tokenized text is split into samples of length `sample_size` and stored in the
           file opened in the prepare step. Here `token_dtype` is used.
            - Trailing tokens will be combined with samples of a next text line
            - Tokens from different text samples will be separated by `eos_idx`
         * If a `sample_size` is not given:
           The tokenized text is immediately stored as is, in the file opened in the prepare step.
           Here `token_dtype` is used.

        :param text_line: JSONL text line

        :return: List of DataSample objects. For each created sample the following is given:
            * Its representation in bytes, as used to store the sample
            * The file path to where the sample is stored

        """

        input_text = self._process_text_line_func(text_line)
        tokenized_text = self._tokenizer_func(input_text)

        if self._sample_size is not None:
            # Always append EOS after each document to mark document boundary
            tokenized_text = tokenized_text + [self._eos_idx]

            # Prepend any leftover tokens from previous document
            if self._rest_tokens is not None:
                tokenized_text = self._rest_tokens + tokenized_text
                self._rest_tokens = None

            num_tokens = len(tokenized_text)
            num_samples = num_tokens // self._sample_size
            num_rest_tokens = num_tokens % self._sample_size

            if num_rest_tokens > 0:
                # Store remainder tokens (includes EOS from this document)
                self._rest_tokens = tokenized_text[-num_rest_tokens:]
                tokenized_text = tokenized_text[:num_samples * self._sample_size]

            tokenized_samples = np.array(tokenized_text, dtype=self._token_dtype)
            tokenized_samples = tokenized_samples.reshape(-1, self._sample_size)
        else:
            tokenized_samples = np.array([tokenized_text], dtype=self._token_dtype)

        # Store tokenized_samples
        sample_data = []
        for sample_idx in range(tokenized_samples.shape[0]):
            sample_bytes = tokenized_samples[sample_idx, :].tobytes()
            sample_data.append(SampleData(
                sample_bytes, self._current_samples_path,
            ))

            self._current_samples_file.write(sample_bytes)

        return sample_data

    def finish(self, is_last_file: bool):
        """
        ## finish ##
         * After all text lines are processed, the file holding the tokenized text samples is closed.
           When `sample_size` not None: Any final trailing tokens will be discarded, but only when the last
           input text file was processed.

        :param is_last_file:
        :return:
        """
        self._close_current_samples_file()

        if is_last_file and self._rest_tokens is not None:
            self._log.debug(f"Cut off {len(self._rest_tokens)} unused tokens")

    def _close_current_samples_file(self):
        if self._current_samples_file:
            self._log.debug(f"Closing tokenized samples file: \n{self._current_samples_path}")
            self._current_samples_file.close()
            self._current_samples_file = None

    def __del__(self):
        self._close_current_samples_file()