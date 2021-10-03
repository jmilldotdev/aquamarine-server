from abc import ABC
from abc import abstractmethod
from collections.abc import Generator

from PIL import Image


class Adapter(ABC):
    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def text_in_scope(self) -> Generator[str, None, None]:
        raise NotImplementedError

    @property
    @abstractmethod
    def images_in_scope(self) -> Generator[Image.Image, None, None]:
        raise NotImplementedError
