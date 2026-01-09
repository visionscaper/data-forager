import os
from typing import Optional

from data_forager.index_stores.fs_based import IndexStore as FSBasedIndexStore
from data_forager.indexers.text_lines import FileTextLinesIndexer
from data_forager.utils import find_files_recursive, natural_sort


def create_default_jsonl_indexer(
    input_base_path: str,
    name: Optional[str] = None,
) -> FileTextLinesIndexer:
    """
    Create JSONL file indexer callable, indexing the JSONL samples in each file at input_base_path (recursively).

    Usage:
    # Create indexing function
    index_jsonl_files_func = create_default_jsonl_indexer(input_base_path='.')
    # Start indexing
    index_jsonl_files_func()

    :param input_base_path: Path to directory containing JSONL files (searched recursively).
    :param name: Optional: name of the indexer to create, used for logging purposes

    :return: FileTextLinesIndexer instance that can be used to index the JSONL files at input_base_path (recursively).
    """
    index_store = FSBasedIndexStore(
        base_path=input_base_path,
    )
    input_file_paths = find_files_recursive(
        input_base_path,
        extension_patterns=['*.jsonl', '*.JSONL']
    )

    # Assuming numbered files
    input_file_paths = natural_sort(input_file_paths)

    return FileTextLinesIndexer(
        input_file_paths=input_file_paths,
        index_store=index_store,
        name=name,
    )
