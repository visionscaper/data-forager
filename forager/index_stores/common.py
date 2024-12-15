from typing import Protocol, Optional

from forager.sample_index import SampleIndex


class IndexStoreInterface(Protocol):

    def load(self) -> SampleIndex:
        pass

    def store(self, index: SampleIndex, from_index: Optional[int] = None):
        pass

