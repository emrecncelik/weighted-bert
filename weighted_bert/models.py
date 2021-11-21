import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer


class Weighter:
    """
    Implementation of Weighted Average scheme proposed in,
    Li, Y., Cai, J., & Wang, J. (2020, June).
    A Text Document Clustering Method Based on Weighted BERT Model.
    https://ieeexplore.ieee.org/document/9085059

    Second weighting sceme (Weighted Removal) will be added in the future.
    """

    def __init__(
        self,
        model_name_or_path: str,
        weight_per_entity: float = 1,
        min_weight: float = 1,
    ) -> None:
        """
        Document embedding model that weights sentences in the document
        according to number of entities in each sentence.

        Args:
            model_name_or_path (str): NER or any other token classification model
                from Hugging Face model hub, or local path to model
            weight_per_entity (float, optional): Weight of each weight token (eg. entity)
                found in a sentence. Defaults to 1.
            min_weight (float, optional): Mininum weight of sentences in case some sentences
                have no weight tokens. Also added to sentences that contain weight tokens. Defaults to 1.
        """
        self.model_name_or_path = model_name_or_path
        self.weight_per_entity = weight_per_entity
        self.min_weight = min_weight

        self._load_model()

    def _load_model(self):
        model = AutoModelForTokenClassification.from_pretrained(self.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.weighting_model = pipeline(
            "ner", model=model, tokenizer=tokenizer, grouped_entities=True
        )

    def _get_sentence_weight(self, entity_count: int):
        # Formula (1) from the paper
        return entity_count * self.weight_per_entity + self.min_weight

    def _get_weighted_document(self, weights: List[int], embeddings: np.ndarray):
        # Formula (2) from the paper
        weights = np.array(weights, ndmin=2).T
        return np.sum(embeddings * weights, axis=0) / np.sum(weights)

    def get_document_embedding(self, texts: List[str], embeddings: np.ndarray):
        entities = self.weighting_model(texts)
        weights = [
            self._get_sentence_weight(len(entity_list)) for entity_list in entities
        ]
        document_embedding = self._get_weighted_document(weights, embeddings)
        return document_embedding


class Embedder:
    def __init__(
        self,
        model_name_or_path: str,
        embedding_layer: Union[int, List[int]] = 0,
        sentence_transformer: bool = True,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.embedding_layer = embedding_layer
        self.sentence_transformer = (
            True
            if "sentence-transformer" in self.model_name_or_path
            else sentence_transformer
        )

        self.embedding_model = None

        self._load_model()

    def _load_model(self):
        if self.sentence_transformer:
            self.embedding_model = SentenceTransformer(self.model_name_or_path)
        else:
            raise NotImplementedError()

    def get_sentence_embeddings(self, texts: List[str]):
        if self.sentence_transformer:
            return self.embedding_model.encode(texts)
        else:
            raise NotImplementedError()


class ClusteringModel:
    def __init__(self, model: str) -> None:
        self.model = model


class WeightedEmbeddingClusterer:
    def __init__(
        self,
        weighting_model_name_or_path: str = "savasy/bert-base-turkish-ner-cased",
        embedding_model_name_or_path: str = "dbmdz/bert-base-turkish-cased",
        clustering_algorithm: str = "kmeans",
        weight_per_token: float = 1,
        min_weight: float = 1,
    ) -> None:
        self.weighting_model_name_or_path = weighting_model_name_or_path
        self.embedding_model_name_or_path = embedding_model_name_or_path
        self.clustering_algorithm = clustering_algorithm
        self.weight_per_token = weight_per_token
        self.min_weight = min_weight


if __name__ == "__main__":
    weighting_model = "savasy/bert-base-turkish-ner-cased"
    embedding_model = "sentence-transformers/distiluse-base-multilingual-cased-v1"

    texts = [
        "Tesla'nın otomobilleri insan hayatlarını riske atıyor olabilir.",
        "Türkiye ve Kore arasında gerçekleşen voleybol müsabakasını Türkiye Milli Takımı kazandı.",
        "Mustafa Kemal Atatürk 19 Mayıs 1919'da Samsun'a ayak bastı.",
        "Bu bir metin.",
    ]

    wm = Weighter(model_name_or_path=weighting_model)
    em = Embedder(model_name_or_path=embedding_model)

    embeddings = em.get_sentence_embeddings(texts)
    weighted_doc = wm.get_document_embedding(texts, embeddings)
