# Weighted BERT

Nonofficial implementation of the paper [A Text Document Clustering Method Based on Weighted BERT Model](https://ieeexplore.ieee.org/document/9085059).

## Installation
```bash
pip install git+https://github.com/emrecncelik/weighted-bert.git
```
## Usage

```python
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from weighted_bert.models import WeightedAverage, WeightedRemoval

logging.basicConfig(level="INFO")

# Select any NER and embedding model from Hugging Face Hub
weighting_checkpoint = "savasy/bert-base-turkish-ner-cased"
embedding_checkpoint = "sentence-transformers/distiluse-base-multilingual-cased-v1"

# Example sentence tokenized documents, normally they should be longer 
documents = [
[
    "Tesla'nın otomobilleri insan hayatlarını riske atıyor olabilir.",
    "Türkiye ve Kore arasında gerçekleşen voleybol müsabakasını Türkiye Milli Takımı kazandı.",
    "Bu bir metin.",
],
[
    "Mustafa Kemal Atatürk 19 Mayıs 1919'da Samsun'a ayak bastı.",
    "Bu bir metin.",
],
]

# Initialize models
embedding_model = SentenceTransformer(embedding_checkpoint)
weighter_a = WeightedAverage(weighting_checkpoint)
weighter_r = WeightedRemoval(weighting_checkpoint)

# Get sentence embeddings of documents
collection_sentence_embeddings = [embedding_model.encode(doc) for doc in documents]

# Calculate weighted embeddings
embeddings_a = np.array([weighter_a.get_document_embedding(doc, sentence_emb)
                for doc, sentence_emb in zip(documents, collection_sentence_embeddings)])
embeddings_r = weighter_r.get_document_embeddings(documents, collection_sentence_embeddings)
```