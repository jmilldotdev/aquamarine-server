from collections.abc import Callable
from collections.abc import Generator
from pathlib import Path
from typing import Optional
from typing import Union

from PIL import Image

from aquamarine import const
from aquamarine.models import Adapter
from aquamarine.models import TextEncoding


class LocalAdapter(Adapter):
    def __init__(
        self,
        path: Union[Path, str],
        text_encoding_type: TextEncoding = TextEncoding.FILE,
        text_preprocessor_fn: Optional[Callable] = None,
        text_blocks_preprocessor_fn: Optional[Callable] = None,
    ) -> None:
        self.path: Path = Path(path)
        self.text_encoding_type: TextEncoding = text_encoding_type
        self.text_preprocessor_fn = text_preprocessor_fn
        self.text_blocks_preprocessor_fn = text_blocks_preprocessor_fn
        super().__init__()

    @property
    def text_in_scope(self) -> Generator[str, None, None]:
        for path in self.path.glob("**/*"):
            if path.suffix in const.TEXT_EXTENSIONS:
                text = path.read_text()
                if self.text_preprocessor_fn:
                    text = self.text_preprocessor_fn(text)
                if self.text_encoding_type == TextEncoding.FILE:
                    yield text
                elif self.text_encoding_type == TextEncoding.BLOCKS:
                    blocks = self.text_blocks_preprocessor_fn(text)
                    for block in blocks:
                        yield block

    @property
    def images_in_scope(self) -> Generator[Image.Image, None, None]:
        for path in self.path.glob("**/*"):
            if path.suffix in const.IMAGE_EXTENSIONS:
                yield Image.open(path).convert("RGB")
