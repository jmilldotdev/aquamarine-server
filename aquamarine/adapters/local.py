from collections.abc import Callable
from collections.abc import Generator
from pathlib import Path
from typing import Optional
from typing import Union

from PIL import Image

from aquamarine import const
from aquamarine.models import Adapter
from aquamarine.models import ContentTypes
from aquamarine.models import ImageContent
from aquamarine.models import TextContent
from aquamarine.models import TextEncoding


class LocalAdapter(Adapter):
    def __init__(
        self,
        path: Union[Path, str],
        text_preprocessor_fn: Optional[Callable] = None,
        text_blocks_preprocessor_fn: Optional[Callable] = None,
    ) -> None:
        self.path: Path = Path(path)
        self.text_preprocessor_fn = text_preprocessor_fn
        self.text_blocks_preprocessor_fn = (
            text_blocks_preprocessor_fn or self.default_block_splitter
        )
        super().__init__()

    @property
    def text_in_scope(self) -> Generator[TextContent, None, None]:
        count = 0
        for path in self.path.glob("**/*"):
            if path.suffix in const.TEXT_EXTENSIONS:
                text = path.read_text()
                if self.text_preprocessor_fn:
                    text = self.text_preprocessor_fn(text)
                yield TextContent(
                    corpus_id=count,
                    content_type=ContentTypes.TEXT,
                    path=str(path),
                    title=path.stem,
                    text=text,
                    text_encoding=TextEncoding.FILE,
                    embedding=None,
                )
                count += 1

    @property
    def text_blocks_in_scope(self) -> Generator[TextContent, None, None]:
        count = 0
        for path in self.path.glob("**/*"):
            if path.suffix in const.TEXT_EXTENSIONS:
                text = path.read_text()
                if self.text_preprocessor_fn:
                    text = self.text_preprocessor_fn(text)
                blocks = self.text_blocks_preprocessor_fn(text)
                for block in blocks:
                    yield TextContent(
                        corpus_id=count,
                        content_type=ContentTypes.TEXT,
                        path=str(path),
                        title=path.stem,
                        text=block,
                        text_encoding=TextEncoding.BLOCKS,
                        embedding=None,
                    )
                    count += 1

    @property
    def images_in_scope(self) -> Generator[ImageContent, None, None]:
        count = 0
        for path in self.path.glob("**/*"):
            if path.suffix in const.IMAGE_EXTENSIONS:
                image = Image.open(path).convert("RGB")
                yield ImageContent(
                    corpus_id=count,
                    content_type=ContentTypes.IMAGE,
                    path=str(path),
                    title=path.stem,
                    image=image,
                    embedding=None,
                )
                count += 1

    def default_block_splitter(self, text: str) -> Generator[str, None, None]:
        for line in text.splitlines():
            yield line
