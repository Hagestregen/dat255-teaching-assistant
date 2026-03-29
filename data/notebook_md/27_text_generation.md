<!-- source: 27_text_generation.ipynb -->

# Text generation with transformers

In this notebook we train a decoder-only model that can run in generative mode, just like modern LLMs. We will however train it on a rather specific type of text -- the IMDb reviews we have been classifiying in the past. Now we will not be classifying anything, but rather generate new reviews.

The text generation is an _autoregressive_ process, and there are different strategies one can inplement in order to obtain natural-looking text. We will try to implement several ones, and see how they compare.


```python
import tensorflow as tf
import keras
import tensorflow_datasets
from string import punctuation
```


## Load data


```python
dataset, info = tensorflow_datasets.load(
    'imdb_reviews',
    with_info=True,
    as_supervised=True,
    split=['train', 'test']
)

train_ds, test_ds = dataset[0], dataset[1]
```


Quick check:


```python
for example, label in train_ds.take(1):
  print('text: ', example)
  print('label: ', label.numpy())
```


## Configuration

We need to make som choices on hyperparameters and sequence lengths. You can change these if you like.


```python
vocab_size = 20000  # Only consider the top 20k words
sequence_length = 80  # Max sequence size
embed_dim = 256  # Embedding size for each token
num_heads = 2  # Number of attention heads
feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer
```


## Text vectorisation

The usual process:


```python
def custom_standardization(input_string):
    """Remove html line-break tags and handle punctuation"""
    lowercased = tf.strings.lower(input_string)
    stripped_html = tf.strings.regex_replace(lowercased, "<br />", " ")
    return tf.strings.regex_replace(stripped_html, f"([{punctuation}])", r" \1")
```


```python
text_vectorization = keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size - 1,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
)

text_only_ds = train_ds.map(lambda x, y: x)

text_vectorization.adapt(text_only_ds)
vocabulary = text_vectorization.get_vocabulary()
```


## Prepare the dataset

We want our decoder to predict the next token of the input sentence -- hence our labels will be the next true token in the sentence.

Create a dataset where the labels are shifted by one position.


```python
def prepare_lm_inputs_labels(text, labels):
    """
    Shift word sequences by 1 position so that the target for position (i) is
    word at position (i+1). The model will use all words up till position (i)
    to predict the next word.

    Discard the original labels, which we don't need.
    """
    #text = tf.expand_dims(text, -1)
    tokenized_sentences = text_vectorization(text)
    x = tokenized_sentences[:-1]
    y = tokenized_sentences[1:]
    return x, y

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(prepare_lm_inputs_labels, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(prepare_lm_inputs_labels, num_parallel_calls=AUTOTUNE)
```


Batching and prefetching


```python
batchsize = 64

train_ds = train_ds.batch(batchsize).prefetch(AUTOTUNE)
test_ds = test_ds.batch(batchsize).prefetch(AUTOTUNE)
```


Verify that the targets are in fact the original sequence, but shifted one position to the right:


```python
for example, label in train_ds.take(1):
    print('text.shape:', example.shape)
    print('text: ', example[0].numpy())
    print('label: ', label[0].numpy())
```


## Model components

We need positional embeddings, and we need a transformer decoder.


```python
class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = keras.layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = keras.layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.not_equal = keras.layers.Lambda(lambda x: tf.math.not_equal(x, 0))

    def call(self, inputs):
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(self.positions)
        return embedded_tokens + embedded_positions

    def build(self, input_shape):
        length = input_shape[-1]
        self.positions = tf.range(start=0, limit=length, delta=1)

    def compute_mask(self, inputs, mask=None):
        return self.not_equal(inputs)

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config


class TransformerDecoder(keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = keras.layers.MultiHeadAttention(
          num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = keras.layers.MultiHeadAttention(
          num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [keras.layers.Dense(dense_dim, activation="relu"),
             keras.layers.Dense(embed_dim),]
        )
        self.layernorm_1 = keras.layers.LayerNormalization()
        self.layernorm_2 = keras.layers.LayerNormalization()
        self.layernorm_3 = keras.layers.LayerNormalization()
        self.supports_masking = True

    def get_config(self):
        config = super(TransformerDecoder, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1),
             tf.constant([1, 1], dtype=tf.int32)], axis=0)
        return tf.tile(mask, mult)

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(
                mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)
        else:
            padding_mask = mask
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=causal_mask)
        attention_output_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(
            query=attention_output_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        attention_output_2 = self.layernorm_2(
            attention_output_1 + attention_output_2)
        proj_output = self.dense_proj(attention_output_2)
        return self.layernorm_3(attention_output_2 + proj_output)
```


## Define the model

We set up our model to output the logits, and not the score after softmax, so that we can add temperature scaling to the softmax later.

In this case we need to match out


```python
embed_dim = 256
latent_dim = 2048
num_heads = 2

inputs = keras.Input(shape=(sequence_length,), dtype="int64")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, x)
outputs = keras.layers.Dense(vocab_size, activation=None)(x)    # no softmax, apply it later
model = keras.Model(inputs, outputs)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="rmsprop"
)
```


Train the model:


```python
model.fit(train_ds, epochs=15, validation_data=test_ds)
```


## Generate text


Approach 1: Select most probable token.


```python
import numpy as np

def most_probable(predictions):
    """
    Return index of the most probable token
    """
    return np.argmax(predictions)
```


Get token indices from vocabulary


```python
tokens_index = dict(enumerate(text_vectorization.get_vocabulary()))
```


```python
prompt = "This movie"
generate_length = 50

sentence = prompt
for i in range(generate_length):
    tokenized_sentence = text_vectorization([sentence])[:, :sequence_length]
    predictions = model(tokenized_sentence)
    next_token = most_probable(
        predictions[0, i, :]
    )
    sampled_token = tokens_index[next_token]
    sentence += " " + sampled_token
print(sentence)
```


Approach 2:

### <span style="color: red;">Exercise:<span>

Implement top-K sampling.

For the sampling itself (after selecting the top token scores), you can use

```
samples = np.random.multinomial(1, predictions, 1)
return np.argmax(samples)
```

([NumPy docs](https://numpy.org/doc/2.2/reference/random/generated/numpy.random.multinomial.html))


```python
# Your code
```


Approach 3:

### <span style="color: red;">Exercise:<span>:

Compute token scores using softmax with temperature.

The equation is

$$
y = \frac{\exp(a_i / T)}{\sum_j \exp(a_j /T)} \,,
$$

where $T$ is the temperature, $a_i$ is the logit of the token in question.


```python
# Your code
```


Approach 4:

### <span style="color: red;">Exercise:<span> (more difficult)

Implement beam search. For this you will need to manage several (let's say 3 to 5) parallel branches of outputs up to a certain length, and then compute the probabilities of each branch, before selecting the most likely one:

![](https://d2l.ai/_images/beam-search.svg)

For more information about beam search, have a look at https://d2l.ai/chapter_recurrent-modern/beam-search.html, or other sources.

