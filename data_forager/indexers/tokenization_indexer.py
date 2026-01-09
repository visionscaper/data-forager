from typing import Callable, List, Optional

import os
from pathlib import Path

import json

import numpy as np

from basics.base import Base

from data_forager.index_stores.fs_based import IndexStore as FSBasedIndexStore
from data_forager.indexers.text_lines import SampleData, FileTextLinesIndexer
from data_forager.utils import find_files_recursive, natural_sort

TokenizerFunc = Callable[[str], List[int]]
ProcessTextLineFunc = Callable[[bytes], str]


def get_text_from_jsonl(jsonl_bytes: bytes, text_key: str = "text", text_encoding: str = "utf-8") -> str:
    jsonl_text = jsonl_bytes.decode(text_encoding)
    data = json.loads(jsonl_text)
    return data[text_key]


def create_tokenize_and_index_jsonl_text_func(
    input_base_path: str,
    tokenizer_func: TokenizerFunc,
    eos_idx: int,
    process_text_line_func: Optional[ProcessTextLineFunc] = None,
    name: Optional[str] = None,
    **sample_generator_kwargs,
) -> FileTextLinesIndexer:
    """
    Create function to:
     * Tokenize text from input JSONL objects, loaded from files at input_base_path (recursively),
     * Store the token data in bin files under folder "tokenized-samples" in input_base_path
     * Store index data under folder "index" in input_base_path

    Usage:
    # Create pipeline to tokenize text from input JSONL objects and index the token samples
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    def tokenize_text(text: str) -> List[int]:
        return tiktoken.enc.encode_ordinary(text)

    tokenize_and_index_jsonl_text_func = create_jsonl_text_tokenization_and_indexing_pipeline(
        input_base_path='.',
        tokenizer_func=tokenize_text,
        sample_size=1024
    )

    # Start tokenization and indexing
    tokenize_and_index_jsonl_text_func()

    :param input_base_path: Path to directory containing JSONL files (searched recursively).
    :param tokenizer_func: Function used to tokenize text.
    :param eos_idx: EOS token index, known by the used Tokenizer
    :param process_text_line_func: Function used to process text lines.
        By default, this converts input JSON lines to dicts and returns the "text" field.
        See function get_text_from_jsonl().
    :param sample_generator_kwargs: Other kwargs passed to TokenizedSampleGenerator.
    :param name: Optional: name of the indexer to create, used for logging purposes

    :return: FileTextLinesIndexer instance that can be used to tokenize and index text from jsonl objects, from
             JSONL files at input_base_path (recursively)
    """
    if process_text_line_func is None:
        process_text_line_func=get_text_from_jsonl

    index_store = FSBasedIndexStore(
        base_path=input_base_path,
    )
    input_file_paths = find_files_recursive(
        input_base_path,
        extension_patterns=['*.jsonl', '*.JSONL']
    )

    # Assuming numbered files
    input_file_paths = natural_sort(input_file_paths)

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


class TokenizedSampleGenerator(Base):

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
            if self._rest_tokens is not None:
                tokenized_text = self._rest_tokens + tokenized_text
                self._rest_tokens = None

            num_tokens = len(tokenized_text)
            num_samples = num_tokens // self._sample_size
            num_rest_tokens = num_tokens % self._sample_size

            if num_rest_tokens > 0:
                self._rest_tokens = tokenized_text[-num_rest_tokens:] + [self._eos_idx]
                tokenized_text = tokenized_text[:num_samples*self._sample_size]

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