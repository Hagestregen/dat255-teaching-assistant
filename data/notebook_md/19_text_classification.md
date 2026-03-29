<!-- source: 19_text_classification.ipynb -->

# Text classification

In this notebook we train networks to do _sentiment analysis_, where we will classify film reviews as either positive or negative.

A popular dataset for this type of task is the [IMDb movie review](https://huggingface.co/datasets/stanfordnlp/imdb) dataset, which is described in detail in this [article](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf). It contains in total 50 000 film reviews from the IMDb website, where users write short reviews accompanied by a rating. On basis of the rating, reviews are categorised into positive or negative, and it is our job to match the text to the class. The data are split 50/50 into 25 000 training examples, and 25 000 testing examples.

We will try out a set of different models -- your regular `Dense` feed-forward network, a CNN, and an RNN. Crucial to training any model at all, is of course to convert the text into numbers. This is the **text vectorisation** step, which we will test out different approaches for here.


Imports


```python
import os
import re
import shutil
import string
import keras
import tensorflow as tf
```


## Download the data


```python
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file(
    "aclImdb_v1",
    url,
    untar=True,
    cache_dir='.',
    cache_subdir=''
)

dataset_dir = 'aclImdb_v1/aclImdb'
```


Let's list the contents of the dataset directory. It contains one directory with positive reviews (`pos`), one with negative (`neg`), and one for unsupervised learning (`unsup`):


```python
train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)
```


We will not use the set for unsupervised learning, so this we can delete.


```python
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)
```


## Create TensorFlow datasets

The movie reviews are stored as separate files in the `pos` and `neg` directories, and can be read into TensorFlow datasets very easily by using Keras' convenience functions.


```python
batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    os.path.join(dataset_dir, 'train'),
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed
)
```


Let's have a look at the first three reviews:


```python
for text_batch, label_batch in raw_train_ds.take(1):
  for i in range(3):
    print("Review", text_batch.numpy()[i])
    print("Label", label_batch.numpy()[i])
    print()
```


Verify the mapping between label and class:


```python
print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])
```


Now we do the same for validation and test datasets.


```python
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    os.path.join(dataset_dir, 'train'),
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed
)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    os.path.join(dataset_dir, 'test'),
    batch_size=batch_size
)
```


## Text vectorisation

Now for the essential part: Getting our text into useful numbers. For this we will use the [`TextVectorization`](https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/) layer.

Our text is, however, not entirely ready for processing yet. if you look at the example reviews, they contain certain HTML tags (like `<br /><br />`), and potentially problematic punctuation. We need to write a function that will _standardise_ the text.


### <span style="color: red;">Exercise:<span>

Complete the function below. If should
- Convert all charachters to lower-case
- Remove `<br>` tags and similar
- Remove commas, periods, and other punctuation

For hints, have a look at the lecture notes. You may find the functions `tf.strings.lower` and `tf.strings.regex_replace` useful.


```python
def custom_standardization(input_data):
  lowercase = ...
  without_html = ...
  without_punctuation = ...
  return without_punctuation
```


Now we can give our standardisation function as input to the `TextVectorization` layer.


## Vectorisation approach number 1: Text as a _set_

The words in a text obviously has an ordering, but is this really relevant to out classification task?

For our first text vectorisation approach, we just encode the presence of words in a text, and not the ordering. We'll perform _multi-hot_ encoding, meaning that for an input text, we output a vector where each column represents a word in the vocabulary, and 1 indicatres the words is present while 0 means it is not.

Consider this little eaxmple of multi-hot text vectorisation:

```{python}
>>> l = keras.layers.TextVectorization(output_mode="multi_hot")
>>> l.adapt(["the cat sat on the mat"])
>>> l.get_vocabulary()
['[UNK]', np.str_('the'), np.str_('sat'), np.str_('on'), np.str_('mat'), np.str_('cat')]
>>> l(["the cat sat on the mat"])
<tf.Tensor: shape=(1, 6), dtype=int64, numpy=array([[0, 1, 1, 1, 1, 1]])>
```

In terms of the N-gram naming, this would be _unigrams_, as we are looking at word sequences of length one.


Let's implement the `TextVectorization` layer on our actual data, using also our `custom_standardization` function.

It is generally a good idea to limit the size of the vocabulary to some tens of thousants of words. While modern LLMs have a vocabulary size from 40k to over 100k, we can do well with 10-20k or less for our classification task. Technically, LLMs have vocabularies that consist of _subwords_, so it doesn't really compare in any case. Anyway, let's also pad the output so we always get the same output length.


```python
max_features = 10000

multihot_vectorize_layer = keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='multi_hot',
    pad_to_max_tokens=True
)
```


The vectorisation layer can (as we have seen with other preprocessing layers before) be used as part of a model, or applied directly to the dataset.
Let's choose the second option this time, so we can investigate the effect of the vectorisation. Technically speaking, this is also brings a performance benefit since in is done asynchronously on the CPU during training.

First, call `adapt()` to learn the vocabulary:


```python
# Extract only the data (and not the labels)
train_text = raw_train_ds.map(lambda x, y: x)

# Adapt
multihot_vectorize_layer.adapt(train_text)
```


We can look at the vocabulary by calling `get_vocabulary()`. Print the ten first entries, along with their indices.


```python
ten_first = multihot_vectorize_layer.get_vocabulary()[:10]

print('index    token')
for i, v in enumerate(ten_first):
    print(f'{i} \t \'{v}\'')
```


Give it a test:


```python
multihot_vectorize_layer(["and this is a test"])
```


Great, it works -- now we can apply it to our TensorFlow datasets, and do caching + prefetching, for better performance.


```python
def multihot_vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return multihot_vectorize_layer(text), label

AUTOTUNE = tf.data.AUTOTUNE

multihot_train_ds = raw_train_ds.map(multihot_vectorize_text)
multihot_train_ds = multihot_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

multihot_val_ds = raw_val_ds.map(multihot_vectorize_text)
multihot_val_ds = multihot_val_ds.cache().prefetch(buffer_size=AUTOTUNE)

multihot_test_ds = raw_test_ds.map(multihot_vectorize_text)
multihot_test_ds = multihot_test_ds.cache().prefetch(buffer_size=AUTOTUNE)
```


### Train a densely-connected model

Having vectorised all our text into an unordered set, it is time to train the first (simple) model: A densely connected network.


```python
dense_model = tf.keras.Sequential([
  keras.Input(shape=(max_features,)),
  keras.layers.Dense(64, activation='relu'),
  keras.layers.Dropout(0.4),
  keras.layers.Dense(1, activation='sigmoid')
])

dense_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['binary_accuracy']
)
dense_model.summary()

epochs = 10

history = dense_model.fit(
    multihot_train_ds,
    validation_data=multihot_val_ds,
    epochs=epochs
)
```


Evaluate on the test set:


```python
print('Loss and accuracy on the test set:')
dense_model.evaluate(multihot_test_ds)
```


Let's test the model on our own film reviews!

To include the text vectorization layer, whihc we now did ourside of the Keras model, we can create a new model object that incorporates it:


```python
model_with_vectorisation = tf.keras.Sequential([
    multihot_vectorize_layer,
    dense_model,
])

model_with_vectorisation.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['binary_accuracy']
)
```


```python
examples = tf.constant([
  "The movie was great!",
  "The movie was okay.",
  "The movie was terrible..."
])

model_with_vectorisation.predict(examples)
```


### <span style="color: red;">Optional exercise:<span>

Train a **bigram** model, but vectorising the text into word pairs using

```{python}
text_vectorization = TextVectorization(
    ngrams=2,
    max_tokens=max_features,
    output_mode="multi_hot",
)
```


```python
# your code
```


## Vectorisation approach number 2: Text as a _sequence_

Dropping the word ordering seems a bit unnatural, so let's keep it, and move to model types that can operate on ordered sequences instead.

Then we need a different word encoding scheme -- we code the words to integers instead. The changes required to the `TextVectorization` layer are small:


```python
sequence_length = 300   # cut the text if longer than this

integer_vectorize_layer = keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)
```


Again, adapt it to the data, and have a look at the vocabulary:


```python
integer_vectorize_layer.adapt(train_text)

ten_first = integer_vectorize_layer.get_vocabulary()[:10]

print('index    token')
for i, v in enumerate(ten_first):
    print(f'{i} \t \'{v}\'')
```


Notice that we got an extra token at the beginning --  index 0, which maps to nothing, which we can interpret as "not a word".


Let's vectorise one review text, and have a look at the output.


```python
def integer_vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return integer_vectorize_layer(text), label

text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[1], label_batch[1]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", integer_vectorize_text(first_review, first_label))
```


This review was shorter than our set output length of 250 tokens, and is padded with index 0, which we mapped to an empty string (''). That makes sense, since there are literally no more words past the end of the text.


For yet aother test, we can do both the encoding and decoding side-by-side:


```python
print('original  encoded    decoded')


first_review_vectorised = integer_vectorize_text(first_review, first_label)
for i in range(20):
    words = tf.strings.split(first_review)[i].numpy()
    vect = first_review_vectorised[0][0][i].numpy()
    outword = integer_vectorize_layer.get_vocabulary()[vect]

    print('{:10} {:5}     {:10}'.format(words.decode(), vect, outword))
```


Finally, when we are happy with the vectorisation, apply it to all the TensorFlow datasets.


```python
integer_train_ds = raw_train_ds.map(integer_vectorize_text)
integer_val_ds = raw_val_ds.map(integer_vectorize_text)
integer_test_ds = raw_test_ds.map(integer_vectorize_text)

integer_train_ds = integer_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
integer_val_ds = integer_val_ds.cache().prefetch(buffer_size=AUTOTUNE)
integer_test_ds = integer_test_ds.cache().prefetch(buffer_size=AUTOTUNE)
```


### Train a convolutional model

We already know two types of models that can process sequence data, the first one being the convolutional network.

### <span style="color: red;">Exercise:<span>

Implement a CNN below, and test it. We have alredy added a `Lambda` layer to make sure the data shape matches what a `Conv1D` layers expects.


```python
cnn_model = keras.Sequential([
    keras.Input(shape=(sequence_length,)),
    keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),

    # Your code
    # ...

    keras.layers.Dense(1, activation="sigmoid")
])

cnn_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['binary_accuracy']
)
cnn_model.summary()

epochs = 10

history = cnn_model.fit(
    integer_train_ds,
    validation_data=integer_val_ds,
    epochs=epochs
)
```


### Train an RNN model

Next, give it a go with a recurrent network, for instance bidirectional LSTM.

In case you don't want to wait out the training, you don't have to :)


```python
lstm_model = keras.Sequential([
    keras.Input(shape=(sequence_length,)),
    keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation="sigmoid")
])

lstm_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['binary_accuracy']
)
lstm_model.summary()

epochs = 10

history = lstm_model.fit(
    integer_train_ds,
    validation_data=integer_val_ds,
    epochs=epochs
)
```


## Improving learning through word embeddings

In case you are underwhelmed by the performance of the CNN and LSTM models, there is luckily a trick that will help us: _Word embeddings_.

This is mainly the topic for the next notebook, but you can try it already here, to see if if improves our LSTM model:


```python
lstm_model = keras.Sequential([
    keras.Input(shape=(sequence_length,)),
    keras.layers.Embedding(input_dim=max_features, output_dim=256),     # hmm!
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation="sigmoid")
])

lstm_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['binary_accuracy']
)
lstm_model.summary()

epochs = 10

history = lstm_model.fit(
    integer_train_ds,
    validation_data=integer_val_ds,
    epochs=epochs
)
```

