from dataclasses import dataclass
from enum import Enum
from typing import Optional

from numpy import ndarray
from PIL import Image


class ContentType(Enum):
    TEXT = "text"
    IMAGE = "image"


class TextEncodingType(Enum):
    FILE = "file"
    BLOCKS = "blocks"


@dataclass
class EmbeddedContent:
    corpus_id: int
    content_type: ContentType
    path: str
    title: str
    embedding: Optional[ndarray]


@dataclass
class ImageContent(EmbeddedContent):
    image: Image.Image


@dataclass
class TextContent(EmbeddedContent):
    text: str
    text_encoding: TextEncodingType

    def __str__(self) -> str:
        return self.text


@dataclass
class QueryResult:
    corpus_id: int
    score: float
    content: EmbeddedContent
