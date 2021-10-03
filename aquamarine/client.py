from sentence_transformers import SentenceTransformer


class AquamarineClient:
    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)
