from tokenize import generate_tokens
from typing import Dict, List, Optional, Union

import json
import os
import shutil
import unittest
import random

from tqdm import tqdm

import numpy as np

import tiktoken

from forager.datasets.tokens import TokensDataset
from forager.indexers.tokenization_indexer import (
    create_tokenize_and_index_jsonl_text_func,
    TokenizerFunc
)
from forager.utils import find_files_recursive, natural_sort


def load_jsonl_data(base_path: str) -> List[Dict]:
    input_file_paths = find_files_recursive(
        base_path,
        extension_patterns=['*.jsonl', '*.JSONL']
    )

    input_file_paths = natural_sort(input_file_paths)

    data = []
    for path in input_file_paths:
        with open(path, 'r') as f:
            while text_line := f.readline():
                data.append(json.loads(text_line))

    return data


def generate_tokens_dataset(
        jsonl_data: List[Dict],
        tokenize_text_func: TokenizerFunc,
        eos_idx: int,
        sample_size: Optional[int] = None,
) -> Union[np.ndarray, List[np.ndarray]]:
    tokens = []
    rest_tokens = None
    for data in tqdm(jsonl_data, desc='Generating tokens'):
        jsonl_text_tokens = tokenize_text_func(data['text'])
        if sample_size is not None:

            if rest_tokens is not None:
                jsonl_text_tokens = rest_tokens + jsonl_text_tokens
                rest_tokens = None

            num_tokens = len(jsonl_text_tokens)
            num_samples = num_tokens // sample_size
            num_rest_tokens = num_tokens % sample_size
            if num_rest_tokens > 0:
                rest_tokens = jsonl_text_tokens[-num_rest_tokens:] + [eos_idx]
                jsonl_text_tokens = jsonl_text_tokens[:num_samples * sample_size]

            tokens += jsonl_text_tokens
        else:
            jsonl_text_tokens = np.array(jsonl_text_tokens)
            tokens += [jsonl_text_tokens]

    if sample_size is not None:
        if rest_tokens is not None:
            print(f'generate_tokens_dataset: Num. rest tokens: {len(rest_tokens)}')

        return np.array(tokens).reshape(-1, sample_size)
    else:
        return tokens


class TestTokenizingIndexingJSONL(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.input_base_path = "../data/neural-ram/long-fineweb-docs/sample-100BT/test/unzipped"
        cls.encoder = tiktoken.get_encoding("gpt2")

    def setUp(self):
        shutil.rmtree(os.path.join(self.input_base_path, 'index'), ignore_errors=True)
        shutil.rmtree(os.path.join(self.input_base_path, 'tokenized-samples'), ignore_errors=True)

    def test_tokenizing_indexing_jsonl_with_sample_size(self):
        def tokenize_text(text: str) -> List[int]:
            return self.encoder.encode_ordinary(text)

        tokenize_and_index_jsonl_text_func = create_tokenize_and_index_jsonl_text_func(
            input_base_path=self.input_base_path,
            tokenizer_func=tokenize_text,
            eos_idx=self.encoder.eot_token,
            sample_size=1024
        )
        # Start tokenizing and indexing
        tokenize_and_index_jsonl_text_func()

        print("Loading JSONL data directly ...")
        jsonl_data = load_jsonl_data(self.input_base_path)
        print("Creating tokens dataset directly ...")
        direct_data = generate_tokens_dataset(
            jsonl_data,
            tokenize_text_func=tokenize_text,
            eos_idx=self.encoder.eot_token,
            sample_size=1024
        )
        dataset_from_index = TokensDataset.create_from_index_on_filesystem(
            self.input_base_path,
            token_dtype=np.uint16
        )

        self.assertEqual(direct_data.shape[0], len(dataset_from_index))

        random_indices = list(range(len(dataset_from_index)))
        random.shuffle(random_indices)

        for idx in tqdm(random_indices, total=len(random_indices), desc="Test samples"):
            self.assertTrue(bool(np.all(dataset_from_index[idx] == direct_data[idx])))

        # Just to test read speed
        for idx in tqdm(random_indices, total=len(random_indices), desc="Read samples"):
            sample = dataset_from_index[idx]


    def test_tokenizing_indexing_jsonl_no_sample_size(self):
        def tokenize_text(text: str) -> List[int]:
            return self.encoder.encode_ordinary(text)

        tokenize_and_index_jsonl_text_func = create_tokenize_and_index_jsonl_text_func(
            input_base_path=self.input_base_path,
            tokenizer_func=tokenize_text,
            eos_idx=self.encoder.eot_token,
            # sample_size=None
        )
        # Start tokenizing and indexing
        tokenize_and_index_jsonl_text_func()

        print("Loading JSONL data directly ...")
        jsonl_data = load_jsonl_data(self.input_base_path)
        print("Creating tokens dataset directly ...")
        direct_data = generate_tokens_dataset(
            jsonl_data,
            tokenize_text_func=tokenize_text,
            eos_idx=self.encoder.eot_token,
            # sample_size=None
        )
        dataset_from_index = TokensDataset.create_from_index_on_filesystem(
            self.input_base_path,
            token_dtype=np.uint16
        )

        self.assertEqual(len(direct_data), len(dataset_from_index))

        random_indices = list(range(len(dataset_from_index)))
        random.shuffle(random_indices)

        for idx in tqdm(random_indices, total=len(random_indices), desc="Test samples"):
            self.assertTrue(bool(np.all(dataset_from_index[idx] == direct_data[idx])))

        # Just to test read speed
        for idx in tqdm(random_indices, total=len(random_indices), desc="Read samples"):
            sample = dataset_from_index[idx]

if __name__ == '__main__':
    unittest.main()
