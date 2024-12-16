from typing import List, Optional, Dict, Protocol

from dataclasses import dataclass

import numpy as np


@dataclass
class SampleLocation:

    file_path: str
    byte_offset: int
    num_bytes: int


@dataclass
class SampleIndex:

    file_locations: List[str]

    # 2D array <num_samples, 3>
    # The three elements in a row:
    #     file_index : uint64
    #     byte_offset: uint64
    #     num_bytes  : uint64
    sample_locations: np.ndarray

    def __len__(self) -> int:
        return self.sample_locations.shape[0]

    def __getitem__(self, idx: int) -> SampleLocation:
        sample_location = self.sample_locations[idx, :]
        return SampleLocation(
            file_path=self.file_locations[sample_location[0]],
            byte_offset=int(sample_location[1]),
            num_bytes=int(sample_location[2])
        )
