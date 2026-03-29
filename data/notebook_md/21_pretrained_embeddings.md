<!-- source: 21_pretrained_embeddings.ipynb -->

# Using pretrained embeddings

Instead of training our own embedding layers, we can download and use pre-trained ones, much like we could use pretrained computer vision models and fine-tune them, without re-training the first feature-extraction layers. This is particularly useful in cases where we have little training data.

In this case we will use the [GloVe](https://nlp.stanford.edu/projects/glove/) (Global Vectors for Word Representation) embeddings, which are orignally trained in an unsupervised setting. These are from before big LLMs were a thing, but still capture semantic similarity very well. The GloVe embeddings come in different dimensionalities: 50-, 100-, 200- and 300-dimensional.

Modern LLM embeddings are often a lot bigger, typically with 3000-4000 dimensions. The quality of these are listed on the Huggingface [embedding leaderboard](https://huggingface.co/spaces/mteb/leaderboard).
For serious use cases one would choose from these instead, but for our simple example, GloVe will do fine.

Let's again classify IMDb film reviews.


```python
import os
import shutil
import string
import tensorflow as tf
import keras
import numpy as np
```


## Download dataset

This part is the same as for notebook 22.


Download


```python
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = keras.utils.get_file(
    "aclImdb_v1",
    url,
    untar=True,
    cache_dir='.',
    cache_subdir=''
)

dataset_dir = 'aclImdb_v1/aclImdb'

train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)

remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)
```


Create the TensorFlow datasets


```python
batch_size = 32
seed = 42

train_ds = keras.utils.text_dataset_from_directory(
    os.path.join(dataset_dir, 'train'),
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed
)

val_ds = keras.utils.text_dataset_from_directory(
    os.path.join(dataset_dir, 'train'),
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed
)

test_ds = keras.utils.text_dataset_from_directory(
    os.path.join(dataset_dir, 'test'),
    batch_size=batch_size)
```


Our nice text standardisation function


```python
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  without_html = tf.strings.regex_replace(lowercase, '<[^>]*>', ' ')
  without_punctuation = tf.strings.regex_replace(without_html, '[{}]'.format(string.punctuation), '')
  return without_punctuation
```


```python
max_features = 20000
sequence_length = 300   # cut the text if longer than this

vectorize_layer = keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)

vectorize_layer.adapt(train_ds.map(lambda x, y: x))
```


Apply vectorisation, and cache / prefetch data for performance.


```python
train_ds = train_ds.map(lambda x, y: (vectorize_layer(x), y))
val_ds = val_ds.map(lambda x, y: (vectorize_layer(x), y))
test_ds = test_ds.map(lambda x, y: (vectorize_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
```


## Download pretrained word embeddings


The GloVe embeddings come in a 822MB zip file.


```python
!wget https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
!unzip -q glove.6B.zip
```


We will be using the 100-dimensional embeddings. Let's have a look at the file (but truncate the lines since they are very long):


```python
! head -n 20 glove.6B.100d.txt | cut -c 1-100
```


The file consists of space-separated entries, where the first column is the token, and the remaing hundred are the embedding axes.

We read the entire thing into a dictionary, mapping each word to its embedding.


```python
path_to_glove_file = "glove.6B.100d.txt"

embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print(f"Found {len(embeddings_index)} word vectors")
```


## Prepare the embeddings

Now we need to make it match the word list, or rather the word (token) indices, that we have in our data of IMDb film reviews.

Our `Embedding` layer consist of a matrix where the row number `i` contains the embedding vector for token number `i`.

We loop through our entire vocabulary, and set the corresponding embedding vector.


```python
vocabulary = vectorize_layer.get_vocabulary()

num_tokens = len(vocabulary) #  + 2    # +2 for "padding" and "OOV"
embedding_dim = 100
hits = 0
misses = 0

# Initialise matrix to zeros.
embedding_matrix = np.zeros((num_tokens, embedding_dim))

# Loop over known words
for i, word in enumerate(vocabulary):

    # Get embedding vector
    embedding_vector = embeddings_index.get(word)

    # Copy it to the matrix
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        hits += 1

    # Not found? Leave it as zeros.
    # This includes the representation for "padding" and "OOV"
    else:
        misses += 1

print("Converted %d words (%d misses)" % (hits, misses))
```


Great, now we apply this matrix to an `Embedding` layer.

We set `trainable=False`, meaning the embeddings will not be updated during model training.


```python
embedding_layer = keras.layers.Embedding(
    num_tokens,
    embedding_dim,
    trainable=False,
)
embedding_layer.build((1,))
embedding_layer.set_weights([embedding_matrix])
```


## Build the model

### <span style="color: red;">Exercise:<span>

Build a useful model that takes the embedding layer as input.


```python
# your code
```


## Train the model


```python
for batch in train_ds.take(1):
    print(batch[0].shape)
    print(batch[1].shape)
```


```python
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["binary_accuracy"]
)
```


```python
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)
```


```python
loss, accuracy = model.evaluate(test_ds)
print('Test set accuracy:', accuracy)
```

