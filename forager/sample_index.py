from dataclasses import dataclass
from typing import List


@dataclass
class SampleLocation:

    # file path can be found using SampleIndex.file_locations[file_index]
    file_index: int
    byte_offset: int


@dataclass
class SampleIndex:

    num_samples: int

    file_locations: List[str]

    sample_locations: List[SampleLocation]
