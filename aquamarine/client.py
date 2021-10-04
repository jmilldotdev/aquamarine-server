from typing import Optional
from typing import Union

from numpy import ndarray
from PIL import Image
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from tqdm import tqdm

from aquamarine.models import Adapter
from aquamarine.models import EmbeddedImageContent
from aquamarine.models import EmbeddedTextContent


class AquamarineClient:
    def __init__(
        self,
        adapters: Optional[list[Adapter]] = None,
        image_model_name: str = "clip-ViT-B-32",
        text_model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.adapters = adapters
        self.image_model = SentenceTransformer(image_model_name)
        self.text_model = SentenceTransformer(text_model_name)

    def generate_text_embeddings(self, adapter: Adapter) -> list[EmbeddedTextContent]:
        text_content = []
        for text in tqdm(list(adapter.text_in_scope)):
            text_content.append(self.generate_text_embedding(text))
        return text_content

    def generate_image_embeddings(self, adapter: Adapter) -> list[EmbeddedImageContent]:
        image_content = []
        for image in tqdm(list(adapter.images_in_scope)):
            image_content.append(self.generate_image_embedding(image))
        return image_content

    def generate_text_embedding(self, text: str) -> EmbeddedTextContent:
        embedding = self.embed(text, self.text_model)
        return EmbeddedTextContent(embedding=embedding, text=text)

    def generate_image_embedding(self, image: Image.Image) -> EmbeddedImageContent:
        embedding = self.embed(image, self.image_model)
        return EmbeddedImageContent(embedding=embedding, image=image)

    def embed(
        self,
        content: Union[Image.Image, str],
        model: SentenceTransformer,
    ) -> ndarray:
        return model.encode(content, convert_to_tensor=True, normalize_embeddings=True)

    def query(self, q: str, embedded_content: list[EmbeddedTextContent]):
        qe = self.embed(q, self.text_model)
        embeddings = [content.embedding for content in embedded_content]
        res = util.semantic_search(qe, embeddings, top_k=5)
        return res[0]
