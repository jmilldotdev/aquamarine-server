from collections.abc import Generator
from pathlib import Path
from typing import Union

from PIL import Image

from aquamarine import const
from aquamarine.models import Adapter


class LocalAdapter(Adapter):
    def __init__(self, path: Union[Path, str]) -> None:
        self.path: Path = Path(path)
        super().__init__()

    @property
    def text_in_scope(self) -> Generator[str, None, None]:
        for path in self.path.glob("**/*"):
            if path.suffix in const.TEXT_EXTENSIONS:
                yield path.read_text()

    @property
    def images_in_scope(self) -> Generator[Image.Image, None, None]:
        for path in self.path.glob("**/*"):
            if path.suffix in const.IMAGE_EXTENSIONS:
                yield Image.open(path).convert("RGB")
