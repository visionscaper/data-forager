from typing import Protocol, Any

from forager.sample_index import SampleIndex


class IndexerInterface(Protocol):

    def __call__(self) -> SampleIndex:
        pass

