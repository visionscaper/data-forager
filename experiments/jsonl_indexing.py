import os
import shutil

from forager.indexers.jsonl_indexer import create_default_jsonl_indexer

if __name__ == "__main__":
    input_base_path = "../data/neural-ram/long-fineweb-docs/sample-100BT/test/unzipped"

    shutil.rmtree(os.path.join(input_base_path, 'index'), ignore_errors=True)

    index_jsonl_files_func = create_default_jsonl_indexer(
        input_base_path=input_base_path
    )
    # Start indexing
    index_jsonl_files_func()
