# Weighted BERT

Unofficial implementation of the paper [A Text Document Clustering Method Based on Weighted BERT Model](https://ieeexplore.ieee.org/document/9085059).

The library produces a single embedding per document by weighting each
sentence's embedding by the number of named entities it contains, following
the two schemes proposed in the paper:

- **`WeightedAverage`** — weighted-mean of sentence embeddings (formulas 1
  and 2 in the paper).
- **`WeightedRemoval`** — SIF-style weighted average followed by
  first-singular-vector removal (formulas 3, 4, 6 and 7).

> This tool is **not** validated against the data sets used by the authors.

## Installation

```bash
pip install git+https://github.com/emrecncelik/weighted-bert.git
```

Requires Python 3.8+. Pulls in `transformers`, `sentence-transformers`,
`scikit-learn`, `scipy` and `numpy`.

## User Guide

The pipeline is always the same:

1. Split each document into sentences and encode them with any sentence
   embedding model (the examples use `sentence-transformers`, but any
   encoder that returns a 2-D `numpy` array of shape
   `(n_sentences, dim)` works).
2. Wrap each `(sentences, sentence_embeddings)` pair in an `InputExample`.
3. Fit a weighter (`WeightedAverage` or `WeightedRemoval`) on the full
   collection. Both weighters expose the standard scikit-learn API
   (`fit`, `transform`, `fit_transform`) and return one embedding per
   document.

Each weighter needs **one** entity source — either a Hugging Face token
classification checkpoint via `weighting_model_name`, or a Python callable
via `entity_detector`. Passing neither raises `EntityModelNotProvided`.

Note that `WeightedRemoval.fit` performs SVD across the entire corpus, so
fit it on the full document collection rather than one document at a time.

### 1. Using a Hugging Face NER model

```python
from sentence_transformers import SentenceTransformer
from weighted_bert.data import InputExample
from weighted_bert.models import WeightedAverage, WeightedRemoval

embedding_checkpoint = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
weighting_checkpoint = "savasy/bert-base-turkish-ner-cased"

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

# 1) Encode each document's sentences
embedding_model = SentenceTransformer(embedding_checkpoint)
input_examples = [
    InputExample(doc, embedding_model.encode(doc)) for doc in documents
]

# 2) Build the weighters with a HF NER checkpoint
weighter_a = WeightedAverage(weighting_model_name=weighting_checkpoint)
weighter_r = WeightedRemoval(weighting_model_name=weighting_checkpoint)

# 3) One embedding per document — shape: (n_docs, embedding_dim)
embeds_a = weighter_a.fit_transform(input_examples)
embeds_r = weighter_r.fit_transform(input_examples)
```

### 2. Using a custom entity detector

Pass any callable that takes a list of sentences and returns a list of
"entity lists" (one per sentence). The weighters only read
`len(entity_list)`, so the inner items can be of any type.

```python
import re
from sentence_transformers import SentenceTransformer
from weighted_bert.data import InputExample
from weighted_bert.models import WeightedAverage, WeightedRemoval

KEYWORDS = ["tesla", "atatürk", "türkiye"]

def entity_detector(document: list[str]) -> list[list[dict]]:
    return [
        [
            {"text": kw, "start": m.start(), "end": m.end()}
            for kw in KEYWORDS
            for m in re.finditer(kw, sentence.lower())
        ]
        for sentence in document
    ]

embedding_model = SentenceTransformer(
    "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
)
input_examples = [
    InputExample(doc, embedding_model.encode(doc)) for doc in documents
]

# Note: pass the callable as `entity_detector=...`. Without this kwarg the
# weighter would fall through to loading a HF model (or raise if neither
# is given).
weighter_a = WeightedAverage(entity_detector=entity_detector)
weighter_r = WeightedRemoval(entity_detector=entity_detector)

embeds_a = weighter_a.fit_transform(input_examples)
embeds_r = weighter_r.fit_transform(input_examples)
```

### Constructor arguments

`WeightedAverage`

| Argument | Default | Description |
| --- | --- | --- |
| `weighting_model_name` | `None` | HF token-classification checkpoint or local path. |
| `entity_detector` | `None` | Callable: `list[str] -> list[list[Any]]`. |
| `weight_per_entity` | `1` | Multiplier per entity (formula 1). |
| `min_weight` | `1` | Floor weight, prevents zero-weighted sentences. |
| `entity_detector_kwargs` | `{}` | Extra kwargs forwarded to the detector. |

`WeightedRemoval` takes the same arguments plus `a` (default `10`, used
in formula 3) and defaults `min_weight` to `0`.

> `entity_types` is reserved for future use — passing it currently raises
> `NotImplementedError`.

After fitting, `weighter.embeddings_` holds the embeddings produced by
the **first** call to `transform`/`fit_transform`. For new data, use the
return value of `transform(...)` directly.

See [`example.ipynb`](example.ipynb) for a runnable end-to-end notebook.
