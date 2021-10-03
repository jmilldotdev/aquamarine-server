import ast
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer
from sentence_transformers import util

from aquamarine import const
from aquamarine.util import flatten_list


class EncodeType(Enum):
    FILE = 1
    BLOCK = 2


class Scope:
    pass


class TagScope(Scope):
    pass


class PathScope(Scope):
    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def __repr__(self) -> str:
        return f"PathScope({self.path})"

    @property
    def full_path(self) -> Path:
        return const.OBSIDIAN_VAULT_PATH / self.path

    @property
    def md_files(self) -> Generator[Path, None, None]:
        return self.full_path.rglob("*.md")


@dataclass
class NodeLink:
    node_id: int
    path: str


@dataclass
class Node:
    node_id: int
    path: str
    inlinks: list[NodeLink]
    outlinks: list[NodeLink]


class ObsidianAdapter:
    def __init__(
        self,
        scopes: Optional[list[Scope]] = None,
        model_name: str = "paraphrase-MiniLM-L6-v2",
    ) -> None:
        self.scopes = scopes
        self.model = SentenceTransformer(model_name)
        self.blocks = None
        self.embeddings = None

    @property
    def files_in_scope(self) -> list[Generator[Path, None, None]]:
        return flatten_list([scope.md_files for scope in self.scopes])

    def encode_notes(self, how: EncodeType = EncodeType.FILE, files=None, trim_fn=None):
        if not files:
            files = self.files_in_scope
        notes = [file.read_text() for file in files]
        if trim_fn:
            notes = [trim_fn(note) for note in notes]
        blocks = []
        if how == EncodeType.FILE:
            blocks.extend(notes)
        elif how == EncodeType.BLOCK:
            for note in notes:
                content = note.split("\n")
                blocks.extend([c for c in content if len(c) > 0])
        print(f"Encoding {len(notes)} note{'s' if len(notes) > 1 else ''}...")
        self.embeddings = self.model.encode(blocks, convert_to_tensor=True)
        self.blocks = blocks

    def query(self, query):
        qe = self.model.encode(query, convert_to_tensor=True)
        res = util.semantic_search(qe, self.embeddings, top_k=5)
        return res

    @staticmethod
    def format_links_literal(lit: str) -> str:
        return lit.replace("false,", "False,").replace("true,", "True,")

    def get_node_links(self, metadataframe, links):
        links = ast.literal_eval(self.format_links_literal(links))
        node_links = []
        for link in links:
            if link["path"].split(".")[-1] == "md":
                try:
                    node_links.append(
                        NodeLink(
                            **{
                                "node_id": metadataframe[
                                    metadataframe["file.path"] == link["path"]
                                ].index[0],
                                "path": link["path"],
                            },
                        ),
                    )
                except IndexError:
                    print(link)
                    print(f"Could not find node for {link['path']}")
                    break
        return node_links

    def get_nodes(self, metadataframe):
        nodes = [
            Node(
                **{
                    "node_id": idx,
                    "path": row["file.path"],
                    "inlinks": self.get_node_links(metadataframe, row["file.inlinks"]),
                    "outlinks": self.get_node_links(
                        metadataframe,
                        row["file.outlinks"],
                    ),
                },
            )
            for idx, row in metadataframe.iterrows()
        ]
        return nodes
