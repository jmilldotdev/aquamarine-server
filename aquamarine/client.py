import pickle
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from sklearn.manifold import TSNE
from tqdm import tqdm

from aquamarine.adapter import Adapter
from aquamarine.const import DATA_PATH
from aquamarine.models import EmbeddedContent
from aquamarine.models import ImageContent
from aquamarine.models import QueryResult
from aquamarine.models import TextContent


class AquamarineClient:
    def __init__(
        self,
        adapters: Optional[list[Adapter]] = None,
        embeddings: Optional[str] = None,
        image_model_name: str = "clip-ViT-B-32",
        text_model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.adapters = adapters
        if embeddings:
            self.embeddings = self.load_embeddings(embeddings)
        self.image_model = SentenceTransformer(image_model_name)
        self.text_model = SentenceTransformer(text_model_name)

    @property
    def embeddings_df(self) -> pd.DataFrame:
        return pd.DataFrame([content.__dict__ for content in self.embeddings])

    def generate_text_embeddings(self, adapter: Adapter) -> list[TextContent]:
        text_content = []
        for text in tqdm(list(adapter.text_in_scope)):
            text.embedding = self.embed(text.text, self.text_model)
        return text_content

    def generate_text_block_embeddings(self, adapter: Adapter) -> list[TextContent]:
        text_content = []
        for text in tqdm(list(adapter.text_blocks_in_scope)):
            text.embedding = self.embed(text.text, self.text_model)
            text_content.append(text)
        return text_content

    def generate_image_embeddings(self, adapter: Adapter) -> list[ImageContent]:
        image_content = []
        for image in tqdm(list(adapter.images_in_scope)):
            image.embedding = self.embed(image.image, self.image_model)
            image_content.append(image)
        return image_content

    def embed(
        self,
        content: Union[Image.Image, str],
        model: SentenceTransformer,
    ) -> np.ndarray:
        return model.encode(content, convert_to_tensor=True, normalize_embeddings=True)

    def query(
        self,
        q: str,
        model: SentenceTransformer,
        embeddings,
        top_k: int = 5,
    ) -> list[QueryResult]:
        qe = self.embed(q, model)
        embeddings = [content.embedding for content in embeddings]
        res = util.semantic_search(qe, embeddings, top_k=top_k)[0]
        query_results = [
            QueryResult(
                corpus_id=r["corpus_id"],
                score=r["score"],
                content=embeddings[r["corpus_id"]],
            )
            for r in res
        ]
        return query_results

    def save_embeddings(self, embeddings: list[EmbeddedContent], fname: str) -> None:
        pickle.dump(embeddings, open(DATA_PATH / f"{fname}.pkl", "wb"))

    def load_embeddings(self, fname: str) -> list[EmbeddedContent]:
        return pickle.load(open(DATA_PATH / f"{fname}.pkl", "rb"))

    def tsne(self, embeddings: list[EmbeddedContent], save: bool = True) -> np.ndarray:
        tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300, init="pca")
        tsne_pca_results = tsne.fit_transform(
            [np.array(c.embedding) for c in embeddings],
        )
        if save:
            pickle.dump(
                list(zip(embeddings, tsne_pca_results)),
                open(DATA_PATH / "tsne.pkl", "wb"),
            )
        return tsne_pca_results

    def load_tsne(self):
        return pickle.load(open(DATA_PATH / "tsne.pkl", "rb"))
