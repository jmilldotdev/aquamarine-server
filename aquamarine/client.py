from sentence_transformers import SentenceTransformer

from aquamarine.models import Adapter


class AquamarineClient:
    def __init__(
        self,
        adapters: list[Adapter],
        model_name: str = "clip-ViT-B-32",
    ) -> None:
        self.adapters = adapters
        self.model = SentenceTransformer(model_name)

    def generate_text_embeddings(self, adapter: Adapter):
        return [
            self.model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
            for text in adapter.text_in_scope
        ]

    def generate_image_embeddings(self, adapter: Adapter):
        return [
            self.model.encode(image, convert_to_tensor=True, normalize_embeddings=True)
            for image in adapter.images_in_scope
        ]

    def generate_embeddings_for_adapter(self, adapter: Adapter):
        text_embeddings = self.generate_text_embeddings(adapter)
        image_embeddings = self.generate_image_embeddings(adapter)
        return text_embeddings, image_embeddings
