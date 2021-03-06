import pickle
from collections.abc import Callable
from collections.abc import Generator
from pathlib import Path
from typing import Optional
from typing import Union

from PIL import Image

from aquamarine import const
from aquamarine.adapter import Adapter
from aquamarine.models import ContentType
from aquamarine.models import ImageContent
from aquamarine.models import TextContent
from aquamarine.models import TextEncodingType


class LocalAdapter(Adapter):
    def __init__(
        self,
        path: Union[Path, str],
        text_preprocessor_fn: Optional[Callable] = None,
        text_blocks_preprocessor_fn: Optional[Callable] = None,
        alias=None,
        load_content: bool = True,
    ) -> None:
        self.path: Path = Path(path)
        self.active_paths = [self.path]
        self.text_preprocessor_fn = text_preprocessor_fn
        self.text_blocks_preprocessor_fn = (
            text_blocks_preprocessor_fn or self.default_block_splitter
        )
        if alias is None:
            alias = self.path.parts[-1]
        if load_content:
            self.embedded_content = self.load_embedded_content(alias)
        else:
            self.embedded_content = None
        super().__init__(alias=alias)

    @property
    def subpaths(self) -> Generator[Path, None, None]:
        for subdir in self.path.glob("**/"):
            yield subdir

    @property
    def text_in_scope(self) -> Generator[TextContent, None, None]:
        return self.get_items_from_paths(self.active_paths, const.TEXT_EXTENSIONS)

    @property
    def images_in_scope(self) -> Generator[ImageContent, None, None]:
        return self.get_items_from_paths(self.active_paths, const.IMAGE_EXTENSIONS)

    def get_items_from_paths(
        self,
        paths: list[Path],
        extensions: list[str],
    ):
        for ap in paths:
            for path in ap.glob("**/*"):
                if path.suffix in extensions:
                    yield path

    def open_text(self, path: str, corpus_id: int) -> TextContent:
        text = path.read_text()
        if self.text_preprocessor_fn:
            text = self.text_preprocessor_fn(text)
        return TextContent(
            corpus_id=corpus_id,
            content_type=ContentType.TEXT,
            path=str(path),
            title=path.stem,
            text=text,
            text_encoding=TextEncodingType.FILE,
            embedding=None,
        )

    def open_text_as_blocks(
        self,
        path: str,
        corpus_id: int,
    ) -> Generator[TextContent, None, None]:
        text = path.read_text()
        if self.text_preprocessor_fn:
            text = self.text_preprocessor_fn(text)
        blocks = self.text_blocks_preprocessor_fn(text)
        return [
            TextContent(
                corpus_id=corpus_id,
                content_type=ContentType.TEXT,
                path=str(path),
                title=path.stem,
                text=block,
                text_encoding=TextEncodingType.BLOCKS,
                embedding=None,
            )
            for block in blocks
        ]

    def open_image(self, path: str, corpus_id: int) -> ImageContent:
        image = Image.open(path).convert("RGB")
        return ImageContent(
            corpus_id=corpus_id,
            content_type=ContentType.IMAGE,
            path=str(path),
            title=path.stem,
            image=image,
            embedding=None,
        )

    def default_block_splitter(self, text: str) -> Generator[str, None, None]:
        for line in text.splitlines():
            yield line

    def load_embedded_content(self, key: str = None):
        if not key:
            key = self.alias
        return pickle.load(open(const.DATA_PATH / f"{key}.pkl", "rb"))
