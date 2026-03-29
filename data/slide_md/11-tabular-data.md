<!-- source: 11-tabular-data.html -->
<!-- index-title: 11: Monday -->

# DAT255 – DAT255: Deep learning engineering
# DAT255: Deep learning engineering

Lecture 11 – Tabular data

sma@hvl.no

---

## This week

**Today:**

- Deep learning on tabular data
- Data preprocessing with Keras - Feature scaling - Converting text and categorical values - Quick look at *embeddings*

**Thursday:**

- Custom Keras objects
- Some ML experiment monitoring tools
- **Project kickoff** - Ideas - Rules - General info

**Today:**

- Deep learning on tabular data
- Data preprocessing with Keras
  - Feature scaling
  - Converting text and categorical values
  - Quick look at _embeddings_

**Thursday:**

- Custom Keras objects
- Some ML experiment monitoring tools

---

## (Some) types of data

---

## Tabular data

Data that makes sense to put in a table:

`patient_id`
`age`
`visits`
`blood_type`
`diag_code`
`symptoms`

321
28
1
“A”
none
“headache, fatigue”

602
64
4
“AB”
32, 12
“complains about headache, joint pains”

201
62
2
“0”
12
“dizzyness, headache”

491
57
1
“A”
6
“headache, fever”

Compare to e.g. images and time series:

---

## Feature normalisation

**Normalisation:**

Make features look like a normal distribution with mean = 0 and variance = 1

Usual approach:

```
import numpy as np
x_normalised = (x - np.mean(x)) / np.std(x)
```

Provided by

- `sklearn.preprocessing.StandardScaler()`
- `keras.layers.Normalization()`

---

## Feature transformations

---

*[Interactive slide: Scikit-learn scalers]*

---

## Dealing with text and categorical data

`patient_id`
`age`
`visits`
`blood_type`
`diag_code`
`symptoms`

321
28
1
“A”
none
“headache, fatigue”

602
64
4
“AB”
32, 12
“complains about headache, joint pains”

201
62
2
“0”
12
“dizzyness, headache”

491
57
1
“A”
6
“headache, fever”

Encoding categorical data with vector entries, like `diag_code`:

- Option 1: **Ordinal** encoding, with one category for each combination ``` {"none": 0, "32, 12": 1, "12": 2, "6": 3} ``` `->` assumes combination is distinct from the separate entries

- Option 2: “**One-hot**” encoding ``` # patient_id   32  12   6 321         [ 0   0   0 ] 602         [ 1   1   0 ] 201         [ 0   1   0 ] 491         [ 0   0   1 ] ``` `->` model will have to learn correlations between columns

---

## Embeddings

Now the magic✨ is:

The embeddings are **learnt from data**

```
# These values are updated during training
# You can have any dimension you like
{"0": [0.2, 1.1], "A": [1.6, -0.2], "B": [-0.3, 0.7], "AB": [1.4, -0.1]}
```

We can treat the embedding space as an Euclidian space where elements **close** to another **are similar** or somehow related.

For categorical data we call it *entity embeddings*
 (later we’ll look at *word embeddings*)

In Keras:

```
keras.layers.Embedding(input_dim, output_dim)
```

- `input_dim`: length of vocabulary
- `output_dim`: dimensions of embedding space (you choose)

---

## Network architectures for tabular data

A transformer model for tabular data: *TabTransformer*

> *TabTransformer significantly outperforms MLP and recent deep networks for tabular data while matching the performance of tree-based ensemble models (GBDT).*

https://arxiv.org/pdf/2012.06678

---

## The best model for tabular data

https://arxiv.org/pdf/2207.08815
