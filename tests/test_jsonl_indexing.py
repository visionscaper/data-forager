from typing import Dict, List

import json
import os
import shutil
import unittest
import random

from tqdm import tqdm

from forager.datasets.jsonl import JsonlDataset
from forager.indexers.jsonl_indexer import create_default_jsonl_indexer
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


class TestJSONLIndexing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.input_base_path = "../data/neural-ram/long-fineweb-docs/sample-100BT/test/unzipped"

        shutil.rmtree(os.path.join(cls.input_base_path, 'index'), ignore_errors=True)


    def test_jsonl_indexing(self):
        index_jsonl_files_func = create_default_jsonl_indexer(
            input_base_path=self.input_base_path
        )
        # Start indexing
        index_jsonl_files_func()

        print("Loading JSONL data directly ...")
        direct_data = load_jsonl_data(self.input_base_path)
        dataset_from_index = JsonlDataset.create_from_index_on_filesystem(self.input_base_path)

        self.assertEqual(len(direct_data), len(dataset_from_index))

        random_indices = list(range(len(dataset_from_index)))
        random.shuffle(random_indices)

        for idx in tqdm(random_indices, total=len(random_indices), desc="Test samples"):
            self.assertEqual(dataset_from_index[idx], direct_data[idx])


if __name__ == '__main__':
    unittest.main()
