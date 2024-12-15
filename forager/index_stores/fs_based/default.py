from typing import Optional

from basics.base import Base

from forager.sample_index import SampleIndex


class IndexStore(Base):

    def __init__(self, base_path: str, name: Optional[str] = None):
        """

        :param base_path: Base path where the index files are stored.

        :param name: Name of instance, if not provided, the classname will be used
        """

        super().__init__(pybase_logger_name=name)

        self._base_path = base_path

    def load(self) -> SampleIndex:
        # TODO
        pass

    def store(self, index: SampleIndex):
        # TODO
        pass