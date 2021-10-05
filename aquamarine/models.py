from abc import ABC
from abc import abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from numpy import ndarray
from PIL import Image


class ContentTypes(Enum):
    TEXT = "text"
    IMAGE = "image"


class TextEncoding(Enum):
    FILE = "file"
    BLOCKS = "blocks"


@dataclass
class EmbeddedContent:
    corpus_id: int
    content_type: ContentTypes
    path: str
    title: str
    embedding: Optional[ndarray]


@dataclass
class ImageContent(EmbeddedContent):
    image: Image.Image


@dataclass
class TextContent(EmbeddedContent):
    text: str
    text_encoding: TextEncoding

    def __str__(self) -> str:
        return self.text


@dataclass
class QueryResult:
    corpus_id: int
    score: float
    content: EmbeddedContent


class Adapter(ABC):
    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def text_in_scope(self) -> Generator[TextContent, None, None]:
        raise NotImplementedError

    @property
    @abstractmethod
    def images_in_scope(self) -> Generator[ImageContent, None, None]:
        raise NotImplementedError
