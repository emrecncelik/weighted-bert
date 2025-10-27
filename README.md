# Weighted BERT

Unofficial implementation of the paper [A Text Document Clustering Method Based on Weighted BERT Model](https://ieeexplore.ieee.org/document/9085059). This tool is **not** tested with the data sets used by the authors.
## Installation
```bash
pip install git+https://github.com/emrecncelik/weighted-bert.git
```
## Usage

### Using NER Model from HF Transformers
```python
weighting_checkpoint = "savasy/bert-base-turkish-ner-cased"
embedding_checkpoint = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"

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

# Calculate embeddings
input_examples = [InputExample(doc, embedding_model.encode(doc)) for doc in documents]
embeds_a = weighter_a.fit_transform(input_examples)
embeds_r = weighter_r.fit_transform(input_examples)
```



### Using a Custom Entity Detector
```python
import re
from typing import List, Dict, Any

# Example function to detect entities,
# It does not actually matter if you return a 
# list of *dictionaries* or not. Weighters only check the length
# of the sentence_entites list for now.
def detect(sentence: str) -> List[Dict]:
    sentence_entites = [] 
    entity_list = ['tesla', "atatürk", "türkiye"]

    for ent in entity_list:
        matches = re.finditer(ent, sentence.lower())
        indexes = [(match.start(), match.end()) for match in matches]
        if indexes:
            for start, end in indexes:
                sentence_entites.append({"text": ent, "start": start, "end": end})
    
    return sentence_entites

# Function to apply detect function to list of docs
def entity_detector(document: List[str]) -> List[List[Dict]]:
    return [detect(sentence) for sentence in document]


# Initialize models
embedding_model = SentenceTransformer(embedding_checkpoint)
weighter_a = WeightedAverage(weighting_checkpoint)
weighter_r = WeightedRemoval(weighting_checkpoint)

# Calculate embeddings
input_examples = [InputExample(doc, embedding_model.encode(doc)) for doc in documents]
embeds_a = weighter_a.fit_transform(input_examples)
embeds_r = weighter_r.fit_transform(input_examples)
```
