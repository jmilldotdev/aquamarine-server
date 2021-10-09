from abc import ABC
from abc import abstractmethod
from collections.abc import Generator
from typing import Any

from aquamarine.models import ImageContent
from aquamarine.models import TextContent


class Adapter(ABC):
    def __init__(self, alias) -> None:
        self.alias = alias

    @property
    @abstractmethod
    def subpaths(self) -> Generator[Any, None, None]:
        raise NotImplementedError

    @property
    @abstractmethod
    def text_in_scope(self) -> Generator[TextContent, None, None]:
        raise NotImplementedError

    @property
    @abstractmethod
    def images_in_scope(self) -> Generator[ImageContent, None, None]:
        raise NotImplementedError
