import os
import glob
from typing import List


def find_files_recursive(input_path: str, extension_patterns: List[str]) -> List[str]:
    file_paths = []
    for ext_pattern in extension_patterns:
        file_paths += list(glob.glob(os.path.join(input_path, "**", ext_pattern), recursive=True))

    return file_paths