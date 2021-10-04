from abc import ABC
from abc import abstractmethod
from collections.abc import Generator
from enum import Enum

from numpy import ndarray
from PIL import Image


class ContentTypes(Enum):
    TEXT = "text"
    IMAGE = "image"


class EmbeddedContent:
    def __init__(self, content_type: ContentTypes, embedding: ndarray) -> None:
        self.content_type = content_type
        self.embedding = embedding


class EmbeddedImageContent(EmbeddedContent):
    def __init__(self, image: Image.Image, embedding: ndarray) -> None:
        self.image = image
        super().__init__(ContentTypes.IMAGE, embedding)


class EmbeddedTextContent(EmbeddedContent):
    def __init__(self, text: str, embedding: ndarray) -> None:
        self.text = text
        super().__init__(ContentTypes.IMAGE, embedding)

    def __str__(self) -> str:
        return self.text


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
