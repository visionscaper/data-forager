from typing import Optional

import numpy as np

from data_forager.sample_index import SampleIndex
from data_forager.datasets.common import Dataset


class TokensDataset(Dataset):

    def __init__(
            self,
            sample_index: SampleIndex,
            token_dtype: np.uint16,
            name: Optional[str] = None,
            **kwargs,
    ):
        super().__init__(sample_index, name=name, **kwargs)

        self._token_dtype = token_dtype

    def _process_sample(self, sample_bytes: bytes) -> np.ndarray:
        return np.frombuffer(sample_bytes, dtype=self._token_dtype)
