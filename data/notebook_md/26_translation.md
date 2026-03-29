<!-- source: 26_translation.ipynb -->

# Sequence-to-sequence transformer for language translation

In this notebook we will try to do _machine translation_ as a sequence-to-sequence task, where we need both parts of the original transformer model -- the encoder as well as the decoder.

In this case we need translated sentences that can be used for training data. Short English sentences with their respective translations into may different languages can be downloaded from [Anki](https://www.manythings.org/anki/). You can choose which language to train on yourself -- but maybe choose one you have some understanding of, in case you want to evaluate the results.


```python
import numpy as np
import tensorflow as tf
import string
import re
import keras
```


## Download data

EXERCISE:

Use `wget` and `unzip` to download an interesting language file from https://www.manythings.org/anki/.


```python
# Your code
```


Have a look at the file (insert the correct file name):


```python
!head <downloaded file>.txt
```


We see it contains three tab-separated columns: English, the translation, and an attribution. The last column we discard when reading in the data.


```python
text_file = "<downloaded file>.txt"     # Insert correct name
with open(text_file) as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []
for line in lines:
    english, translation, _ = line.split("\t")
    translation = "[start] " + translation + " [end]"
    text_pairs.append((english, translation))
```


Look at some samples, to verify the data is correctly read.


```python
import random
print(random.choice(text_pairs))
```


Now, split the text in test and train sets.


```python
random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples:num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples:]
```


## Vectorise the text

Like before we remove punktuation. In case you are using text for a language with different punctuation than English, you should the extra punctuation characters in `strip_chars` below. (Example: If translating Spanish, add the "¿" character).


```python
strip_chars = string.punctuation
strip_chars = strip_chars.replace("[", "")      # Don't remove brackets, since we use them for [start] and [end] tokens.
strip_chars = strip_chars.replace("]", "")

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
        lowercase, f"[{re.escape(strip_chars)}]", "")

vocab_size = 15000
sequence_length = 20

source_vectorization = keras.layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)
target_vectorization = keras.layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)

train_english_texts = [pair[0] for pair in train_pairs]
train_translated_texts = [pair[1] for pair in train_pairs]

source_vectorization.adapt(train_english_texts)
target_vectorization.adapt(train_translated_texts)
```


Prepare TensorFlow datasets:


```python
batch_size = 64

def format_dataset(eng, tran):
    eng = source_vectorization(eng)
    tran = target_vectorization(tran)
    return ({
        "english": eng,
        "translated": tran[:, :-1],
    }, tran[:, 1:])

def make_dataset(pairs):
    eng_texts, tran_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    stran_texts = list(tran_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, tran_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=4)
    return dataset.shuffle(2048).prefetch(16).cache()

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)
```


Check the shapes:


```python
for inputs, targets in train_ds.take(1):
    print(f"inputs['english'].shape: {inputs['english'].shape}")
    print(f"inputs['translated'].shape: {inputs['translated'].shape}")
    print(f"targets.shape: {targets.shape}")
```


## Test a GRU-based variant (optional)

For comparison, train a bidirectional GRU-based network.

This takes a long time, so you don't really have to.


The encoder part:


```python
embed_dim = 256
latent_dim = 1024

source = keras.Input(shape=(None,), dtype="int64", name="english")
x = keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)(source)
encoded_source = keras.layers.Bidirectional(
    keras.layers.GRU(latent_dim),
    merge_mode="sum"
)(x)
```


The decoder part:


```python
past_target = keras.Input(shape=(None,), dtype="int64", name="translation")
x = keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)(past_target)
decoder_gru = keras.layers.GRU(latent_dim, return_sequences=True)
x = decoder_gru(x, initial_state=encoded_source)
x = keras.layers.Dropout(0.5)(x)
target_next_step = keras.layers.Dense(vocab_size, activation="softmax")(x)

seq2seq_rnn = keras.Model([source, past_target], target_next_step)
```


Train the GRU:


```python
seq2seq_rnn.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

seq2seq_rnn.fit(train_ds, epochs=15, validation_data=val_ds)
```


Try out some translations:


```python
translation_vocab = target_vectorization.get_vocabulary()
translation_index_lookup = dict(zip(range(len(translation_vocab)), translation_vocab))
max_decoded_sentence_length = 20

def decode_sequence(input_sentence):

    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"

    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization([decoded_sentence])
        next_token_predictions = seq2seq_rnn.predict(
            [tokenized_input_sentence, tokenized_target_sentence]
        )
        sampled_token_index = np.argmax(next_token_predictions[0, i, :])
        sampled_token = translation_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence

test_eng_texts = [pair[0] for pair in test_pairs]
for _ in range(10):
    input_sentence = random.choice(test_eng_texts)
    print("-")
    print(input_sentence)
    print(decode_sequence(input_sentence))
```


## The sequence-to-sequence transformer

Let's first build the **decoder**. There's a good bit to things required, so we just give the entire code, but try to understand the different parts.

Unlike the encoder from last week, we will in this case use _causal attention_, meaning that at each step in the sequence, the model can only look backwards, and not forwards, when trying to complete the next token of the translated text. To accomplish this, we create a causal attention _mask_, which is a diagonal matrix where the upper half is 0 and the lower half is 1. This "removes" the end of the sentence, but reveals the next token one at a time, for each step in the sequence.


### Define the decoder


```python
class TransformerDecoder(keras.layers.Layer):

    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):

        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        # We use two attentions heads:
        # One for self-attention
        self.attention_1 = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )
        # And one for cross-attention (taking in the eoncoder output)
        self.attention_2 = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        # The dense layer following the attention heads
        self.dense_proj = keras.Sequential(
            [keras.layers.Dense(dense_dim, activation="relu"),
             keras.layers.Dense(embed_dim),]
        )

        # Layer normalisation after each on the three layers
        self.layernorm_1 = keras.layers.LayerNormalization()
        self.layernorm_2 = keras.layers.LayerNormalization()
        self.layernorm_3 = keras.layers.LayerNormalization()
        self.supports_masking = True

    def get_config(self):
        config = super().get_config()
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
             tf.constant([1, 1], dtype=tf.int32)],
            axis=0
        )

        return tf.tile(mask, mult)

    def call(self, inputs, encoder_outputs, mask=None):

        # First, get the causal attention mask
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(
                mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)
        else:
            padding_mask = mask

        # Compute self-attention on inputs
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=causal_mask
        )
        attention_output_1 = self.layernorm_1(inputs + attention_output_1)

        # Compute cross-attention with encoder outputs
        attention_output_2 = self.attention_2(
            query=attention_output_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        attention_output_2 = self.layernorm_2(
            attention_output_1 + attention_output_2)

        # Dense layer
        proj_output = self.dense_proj(attention_output_2)

        return self.layernorm_3(attention_output_2 + proj_output)
```


### Define the encoder

### <span style="color: red;">Exercise:<span>

Define a `TransformerEncoder` class.

<details>
    <summary>Hint</summary>
    Check the past notebooks we did
</details>


```python
# Your code
```


### Add positional embedding

Her we use what we had from the previous notebooks.


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
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

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
```


### Build the complete model

We can in principle add multiple encoder and decoder blocks, but first we try a more minimal model.

The encoder takes in the English text, while the decoder takes in the translation. The `keras.Input` layer can (as we tried in previous notebooks) select input from a TensorFlow dataset by name.


```python
embed_dim = 256
dense_dim = 2048
num_heads = 8

encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="english")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)

decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="translation")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)
x = keras.layers.Dropout(0.5)(x)
decoder_outputs = keras.layers.Dense(vocab_size, activation="softmax")(x)

transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
```


Time to train it!


```python
transformer.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])

transformer.fit(train_ds, epochs=10, validation_data=val_ds)
```


## Try translating text


```python
translation_vocab = target_vectorization.get_vocabulary()
translation_index_lookup = dict(zip(range(len(translation_vocab)), translation_vocab))
max_decoded_sentence_length = 20

def decode_sequence(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization(
            [decoded_sentence])[:, :-1]
        predictions = transformer(
            [tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = translation_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence


test_eng_texts = [pair[0] for pair in test_pairs]
for _ in range(20):
    input_sentence = random.choice(test_eng_texts)
    print("-")
    print(input_sentence)
    print(decode_sequence(input_sentence))
```

