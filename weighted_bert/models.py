import logging
import numpy as np
from typing import Callable, List
from scipy.sparse.linalg import svds
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class EntityModelNotProvided(Exception):
    """Raised when an entity model (rule based function or HF checkpoint) is not provided to the Weighter classes"""

    pass


class WeighterBase:
    """
    Implementation of Weighted Average scheme proposed in,
    Li, Y., Cai, J., & Wang, J. (2020, June).
    A Text Document Clustering Method Based on Weighted BERT Model.
    https://ieeexplore.ieee.org/document/9085059

    """

    def __init__(
        self,
        weighting_model_name: str = None,
        entity_detector: Callable = None,
        entity_types: List[str] = None,
        entity_detector_kwargs: dict = {},
    ) -> None:
        """
        Document embedding model that weights sentences in the document
        according to number of entities in each sentence.

        Args:
            model_name_or_path (str): NER or any other token classification model
                from Hugging Face model hub, or local path to model
            weight_per_entity (float, optional): Weight of each weight token (eg. entity)
                found in a sentence. Defaults to 1.
        """
        if weighting_model_name is None and entity_detector is None:
            raise EntityModelNotProvided(
                "Provide a weighting_model_name (hf checkpoint) or an entity_detector (custom function) while initializing"
            )

        self.weighting_model_name = weighting_model_name
        self.entity_detector = entity_detector
        self.entity_types = entity_types
        self.entity_detector_kwargs = entity_detector_kwargs

        self.embeddings_ = None
        self.entity_counts_ = None

        if entity_detector is None:
            self._load_model()
        else:
            logger.info(
                "Entity detector function you have provided is being used, not initializing HuggingFace model."
            )

    def _load_model(self):
        logger.info("================ Loading weighting model ================")
        logger.info(f"\t{self.weighting_model_name}")
        model = AutoModelForTokenClassification.from_pretrained(
            self.weighting_model_name
        )
        tokenizer = AutoTokenizer.from_pretrained(self.weighting_model_name)
        self.weighting_model = pipeline(
            "ner", model=model, tokenizer=tokenizer, grouped_entities=True
        )

    def _get_entity_count_by_type(self, document: List[str]) -> List[int]:
        if not self.entity_types:
            if self.entity_detector is not None:
                entity_counts = [
                    len(entity_list)
                    for entity_list in self.entity_detector(
                        document, **self.entity_detector_kwargs
                    )
                ]
            else:
                sentence_entities = self.weighting_model(document)
                entity_counts = [len(entity_list) for entity_list in sentence_entities]
            logger.debug(f"Entity count list for doc: {entity_counts}")
            return entity_counts
        else:
            raise NotImplementedError


class WeightedAverage(WeighterBase):
    """
    Implementation of Weighted Average scheme proposed in,
    Li, Y., Cai, J., & Wang, J. (2020, June).
    A Text Document Clustering Method Based on Weighted BERT Model.
    https://ieeexplore.ieee.org/document/9085059

    """

    def __init__(
        self,
        weighting_model_name: str = None,
        entity_detector: Callable = None,
        entity_types: List[str] = None,
        weight_per_entity: int = 1,
        min_weight: int = 1,
        entity_detector_kwargs: dict = {},
    ) -> None:
        """Weighted average model that can be used to
        calculate entity weighted document embeddings.

        Args:
            weighting_model_name (str): Hugging Face checkpoint of a NER model
            entity_types (List[str], optional): Entity types to include into weighting.
                Defaults to None which uses all types.
            weight_per_entity (int, optional): Multiplier for each entity in the sentence,
                see _calculate_sentence_weight method. Defaults to 1.
            min_weight (int, optional): Minimum weight of a sentence. Defaults to 1.
        """
        super().__init__(
            weighting_model_name=weighting_model_name,
            entity_detector=entity_detector,
            entity_types=entity_types,
            entity_detector_kwargs=entity_detector_kwargs,
        )
        self.weight_per_entity = weight_per_entity
        self.min_weight = min_weight

    def get_document_embedding(
        self,
        document: List[str],
        sentence_embeddings: np.ndarray,
    ):
        entity_counts = self._get_entity_count_by_type(document)
        weights = [self._calculate_sentence_weight(count) for count in entity_counts]
        document_embedding = self._calculate_weighted_embedding(
            weights, sentence_embeddings
        )
        return document_embedding

    def _calculate_sentence_weight(self, entity_count: int):
        # Formula (1) from the paper
        return entity_count * self.weight_per_entity + self.min_weight

    def _calculate_weighted_embedding(
        self, weights: List[int], sentence_embeddings: np.ndarray
    ):
        # Formula (2) from the paper
        weights = np.array(weights, ndmin=2).T
        return np.sum(sentence_embeddings * weights, axis=0) / np.sum(weights)


class WeightedRemoval(WeighterBase):
    """
    Implementation of Weighted Removal scheme proposed in,
    Li, Y., Cai, J., & Wang, J. (2020, June).
    A Text Document Clustering Method Based on Weighted BERT Model.
    https://ieeexplore.ieee.org/document/9085059
    """

    def __init__(
        self,
        weighting_model_name: str = None,
        entity_detector: Callable = None,
        entity_types: List[str] = None,
        a: int = 10,
        entity_detector_kwargs: dict = {},
    ) -> None:
        """Weighted removal model # TODO Add explanation

        Args:
            weighting_model_name (str): Hugging Face checkpoint of a NER model
            entity_types (List[str], optional):  Entity types to include into weighting.
                Defaults to None which uses all types.
            a (int, optional): Parameter proportional to p(s) that is used to ,
                see _calculate_sentence_weight method. Defaults to 10.
        """
        super().__init__(
            weighting_model_name=weighting_model_name,
            entity_detector=entity_detector,
            entity_types=entity_types,
            entity_detector_kwargs=entity_detector_kwargs,
        )
        self.a = a  # TODO Calculate "a" considering max(p(s))?
        self.collection_entity_counts_: List[List[int]] = None
        self.total_entity_count_: int = None
        self.trained: bool = False

    def get_document_embeddings(
        self,
        documents: List[List[str]],
        collection_sentence_embeddings: np.ndarray,
    ):
        logger.info("================ Detecting entities ================")
        self._calculate_collection_entity_counts(documents)

        logger.info(
            "================ Calculating initial document embeddings ================"
        )

        initial_document_embeddings = []
        for i in range(len(documents)):
            # Calculate sentence weight for each sentence in document
            sentence_weights = [
                self._calculate_sentence_weight(entity_count)
                for entity_count in self.collection_entity_counts_[i]
            ]
            # Calculate initial document embedding according to sentence weights
            initial_document_embedding = self._calculate_initial_document_embedding(
                collection_sentence_embeddings[i], sentence_weights
            )
            initial_document_embeddings.append(initial_document_embedding)
        initial_document_embeddings = np.array(initial_document_embeddings)

        logger.info("================ Correcting document embeddings ================")
        logger.info("\tCalculating first singular vector...")
        # Perform SVD to calculate "first singular vector" mentioned in the paper
        self._calculate_singular_vector(initial_document_embeddings)

        # Apply removal and get final embeddings
        logger.info("\tCalculating corrected embeddings...")
        corrected_embeddings = np.array(
            [
                self._calculate_corrected_document_embedding(embed)
                for embed in initial_document_embeddings
            ]
        )
        return corrected_embeddings

    def _calculate_sentence_weight(self, entity_count: int):
        # Formula (3) and (4) from the paper
        ps = entity_count / self.total_entity_count_  # (4)
        return self.a / (self.a - ps)  # (3)

    def _calculate_initial_document_embedding(
        self, document_sentence_embeddings: np.ndarray, sentence_weights: List[float]
    ):
        # Formula (6) from the paper
        weighted_sentence_embeddings = (
            document_sentence_embeddings * np.array(sentence_weights, ndmin=2).T
        )
        initial_document_embedding = (
            sum(weighted_sentence_embeddings) / weighted_sentence_embeddings.shape[0]
        )
        return initial_document_embedding

    def _calculate_corrected_document_embedding(
        self, initial_document_embedding: np.ndarray
    ):
        # Formula (7) from the paper
        corrected_embedding = (
            initial_document_embedding - self.u @ self.u.T @ initial_document_embedding
        )
        return corrected_embedding

    def _calculate_singular_vector(self, initial_document_embeddings: np.ndarray):
        u, _, _ = svds(initial_document_embeddings.T, 1)
        self.u = u

    def _calculate_collection_entity_counts(self, documents: List[str]):
        self.collection_entity_counts_ = [
            self._get_entity_count_by_type(doc) for doc in documents
        ]
        self.total_entity_count_ = sum(
            [sum(sentence_counts) for sentence_counts in self.collection_entity_counts_]
        )
