<!-- source: 20_word_embeddings.ipynb -->

# Word embeddings

_Word embeddings_ is a trick that significantly improves the performance of NLP models, and all modern LLMs rely on this. In this notebook we will try to get an understanding of how they work.

Like for the last notebook, let us use the IMDb movie review dataset to do sentiment analysis.


Load TensorBoard, which we will use for visualisation.


```python
%load_ext tensorboard
```


Imports


```python
import os
import string
import keras
import tensorflow as tf
import tensorflow_datasets
from tensorboard.plugins import projector
```


## Download the data


For an easier data loading process, we can download the data directly into TensorFlow using the [TensorFlow datasets](https://www.tensorflow.org/datasets) extension.

You can find several intereseting datasets readily available here, at the expense of being somewhat cumbersome to look at and understand, since they are already `Tensor`s. Anyway, here goes.


```python
dataset, info = tensorflow_datasets.load(
    'imdb_reviews',
    with_info=True,
    as_supervised=True
)
train_ds, test_ds = dataset['train'], dataset['test']
```


Look at the first review:


```python
for example, label in train_ds.take(1):
  print('text: ', example.numpy())
  print('label: ', label.numpy())
```


## Text vectorisation

Like before, we remove punctuation, split words on whitespace, and remove any pesky HTML tags.


```python
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  without_html = tf.strings.regex_replace(lowercase, '<[^>]*>', ' ')
  without_punctuation = tf.strings.regex_replace(without_html, '[{}]'.format(string.punctuation), '')
  return without_punctuation
```


Instantiate and adapt the `TextVectorization` layer:


```python
max_features = 10000
sequence_length = 300   # cut the text if longer than this

vectorize_layer = keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)

vectorize_layer.adapt(
    train_ds.map(lambda x, y: x)
)
```


...and apply to the TensorFlow datasets.

In this case we also need to expand the dimensions of both the data and the label, which is an annoyance you'll get used to.


```python
def vectorize_text(text, label):
  label = tf.expand_dims(label, -1)
  #text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label
```


```python
batch_size = 64

train_ds = train_ds.map(lambda x, y: vectorize_text(x, y))
test_ds = test_ds.map(lambda x, y: vectorize_text(x, y))

train_ds = train_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
```


We can look at the vocabulary by calling `get_vocabulary()`. Print the ten first entries, along with their indices.


```python
ten_first = vectorize_layer.get_vocabulary()[:10]

print('index    token')
for i, v in enumerate(ten_first):
    print(f'{i} \t \'{v}\'')
```


## Define the model

The crucial part of our model will be the `Embedding` layer, which encodes the token indices into a vector of floating-point values. We are free to define the dimensions of the embedding ourselves.

However, as explained in the textbook on page 471, using an embedding dimension that is larger than the number of units in the preceeding layer, is not very useful.

For the rest of the model, we keep it simple.


```python
embedding_dim = 32

model = tf.keras.Sequential([
    keras.Input(shape=(sequence_length,)),
    keras.layers.Embedding(max_features, embedding_dim),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.L2(0.01)),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(1, activation='sigmoid')
])
```


```python
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['binary_accuracy']
)
```


```python
model.summary()
```


Train!


```python
epochs = 10
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=epochs
)
```


```python
loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)
```


If we like, we can write some reviews ourselves and test.


```python
model_with_vectorisation = tf.keras.Sequential([
  vectorize_layer,
  model,
])

model_with_vectorisation.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['binary_accuracy']
)
```


```python
examples = tf.constant([
  "It was the best movie in the history of movies, maybe ever. "
])

pred = model_with_vectorisation.predict(examples)
print('Review was {:.3f}% positive.'.format(pred[0][0]))
```


## Visualise the embedding space

To make any sence of the positions of the different words in embedding space, let's plot it using TensorBoard.

The code below writes two pieces of information to files that TensorBoard can read:
- The list of words in the vocabulary (goes in `metadata.tsv`)
- The weights of the embedding layer.


```python
log_dir='/logs/imdb/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Write one vocabulary entry per line
with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:

  for subwords in vectorize_layer.get_vocabulary():
    f.write("{}\n".format(subwords))

# Save the weights we want to analyze as a variable. Note that the first
# value represents "no word", which we remove.
weights = tf.Variable(model.layers[0].get_weights()[0][1:])

# Create a checkpoint from embedding, the filename and key are the
# name of the tensor
checkpoint = tf.train.Checkpoint(embedding=weights)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

# Set up config.
config = projector.ProjectorConfig()
embedding = config.embeddings.add()

# The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(log_dir, config)
```


Check the metadata file:


```python
! wc -l /logs/imdb/metadata.tsv
!head /logs/imdb/metadata.tsv
```


### Run TensorBoard

To show the embeddings, click the dropdown menu in the upper right and select PROJECTOR.

Now you can hover over the different words, and click them to find neighbouring words.


### <span style="color: red;">Exercise:<span>

Investigate the embeddings and see if you find that the neighbours are in fact related, and have a meaningful position in the embedding space.

Remember that our embedding space was 32-dimensional, which is hard to visualise since we are limited to living in 3-dimensional space. TensorBoard will help us by projecting everything down to 3 dimensions, which means that for the plot we see, a lot of information is lost. Use the distance measures on the right to guide you when comparing words.

If you train for longer or create a better performing model, how does the embedding space change?


```python
%tensorboard --logdir /logs/imdb/
```

