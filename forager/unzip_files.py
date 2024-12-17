import glob
from logging import Logger
from typing import Optional, List

import os
import argparse

import gzip

from basics.logging import get_logger

from forager.utils import find_files_recursive

module_logger = get_logger(os.path.basename(__file__))


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Unzip files')

    parser.add_argument(
        '--input-files-path',
        type=str, required=True,
        help='Path to the GZIPed files.')

    parser.add_argument(
        '--output-path',
        type=str, required=False,
        help='Path to where the unzipped files will be saved. '
             'If not provided, the unzipped files will be saved in a new "unzipped" folder of the given file path.'
    )

    return parser


def get_gzip_ext_pos(zipped_file_name: str) -> int:
    for ext in ['.gz', '.gzip', '.GZ', '.GZIP']:
        try:
            return zipped_file_name.index(ext)
        except ValueError:
            pass

    return -1


def get_unzipped_file_name(zipped_file_path):
    zipped_file_name = os.path.basename(zipped_file_path)
    gzip_ext_pos = get_gzip_ext_pos(zipped_file_name)
    if gzip_ext_pos == -1:
        return f"{zipped_file_name}-unzipped"
    else:
        return zipped_file_name[:gzip_ext_pos]


def unzip_file(
    input_file_path: str,
    output_path: Optional[str] = None,
    logger: Optional[Logger] = None,
) -> None:
    if output_path is None:
        output_path = os.path.dirname(input_file_path)

    if logger is None:
        logger = module_logger

    input_file_name = os.path.basename(input_file_path)
    output_file_name = get_unzipped_file_name(input_file_name)
    output_file_path = os.path.join(output_path, output_file_name)

    logger.debug(f"Uncompress: \n"
                 f"From: {input_file_path}\n"
                 f"To  : {output_file_path}\n")
    with gzip.open(input_file_path, 'rb') as input_file:
        with open(output_file_path, 'wb') as output_file:
            output_file.write(input_file.read())


def find_zipped_files(input_path: str) -> List[str]:
    return find_files_recursive(input_path, extension_patterns=['*.gz', '*.gzip', '*.GZ', '*.GZIP'])


def unzip_files(input_files_path: str, output_path: Optional[str] = None) -> None:

    if output_path is None:
        output_path = os.path.join(input_files_path, 'unzipped')

    os.makedirs(output_path, exist_ok=True)

    input_file_paths = find_zipped_files(input_files_path)
    for input_file_path in input_file_paths:
        unzip_file(input_file_path, output_path)


if __name__ == '__main__':

    parser = create_arg_parser()

    args = parser.parse_args()

    unzip_files(**vars(args))
