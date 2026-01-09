import json
from typing import Optional, Dict

from data_forager.datasets.common import Dataset
from data_forager.sample_index import SampleIndex


class JsonlDataset(Dataset):

    def __init__(
            self,
            sample_index: SampleIndex,
            text_encoding: str = "utf-8",
            name: Optional[str] = None,
            **kwargs,
    ):
        super().__init__(sample_index, name=name, **kwargs)

        self._text_encoding = text_encoding

    def _process_sample(self, sample_bytes: bytes) -> Dict:
        jsonl_text = sample_bytes.decode(self._text_encoding)
        return json.loads(jsonl_text)
