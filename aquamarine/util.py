from typing import Any


def flatten_list(lst: list[Any]) -> list[Any]:
    return [i for s in lst for i in s]
