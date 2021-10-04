import mimetypes
from collections.abc import Generator
from typing import Any


def get_extensions_for_type(general_type: str) -> Generator[str, None, None]:
    mimetypes.init()
    for ext in mimetypes.types_map:
        if mimetypes.types_map[ext].split("/")[0] == general_type:
            yield ext


def format_extensions_for_glob(extensions: list[str]) -> str:
    return "*." + "".join([f"[{ext.split('.')[1]}]" for ext in extensions])


def flatten_list(lst: list[Any]) -> list[Any]:
    return [i for s in lst for i in s]
