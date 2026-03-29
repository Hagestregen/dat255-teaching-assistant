<!-- source: 24_transformer_encoder_for_classification.ipynb -->

# Transformer encoder for classification

We revisit the IMDb sentiment analysis dataset, but now try out the famed Transformer. Since this is a sequence-to-vector task (and not sequence-to-sequence), we need only one part of the proposed architecture, which is the encoder. We will use the encoder to make (hopefully good) feature subspaces, and put a classification layer on top.


https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/chapter11_part03_transformer.ipynb


Imports


```python
import tensorflow as tf
import keras
import tensorflow_datasets
```


## Load and vectorise the data

We load the IMDb movie review data through TensorFlow Datasets, for convenience.


```python
dataset, info = tensorflow_datasets.load(
    'imdb_reviews',
    with_info=True,
    as_supervised=True,
    split=['train[:80%]', 'train[80%:]', 'test']
)

train_ds, val_ds, test_ds = dataset[0], dataset[1], dataset[2]
```


```python
for example, label in train_ds.take(1):
  print('text: ', example.numpy())
  print('label: ', label.numpy())
```


Let's vectorise the data in the usual fashion.


```python
max_length = 600
max_tokens = 20000
text_vectorization = keras.layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_length,
)

data_without_labels = train_ds.map(lambda x, y: x)

text_vectorization.adapt(data_without_labels)
```


Apply the vectorisation layer to the datasets.


```python
def vectorise(inputs):
    x = text_vectorization(inputs)
    return x

int_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
int_val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
int_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
```


Batch and prefetch, for performance:


```python
batchsize = 32
AUTOTUNE = tf.data.AUTOTUNE

int_train_ds = int_train_ds.batch(batchsize).prefetch(AUTOTUNE)
int_val_ds = int_val_ds.batch(batchsize).prefetch(AUTOTUNE)
int_test_ds = int_test_ds.batch(batchsize).prefetch(AUTOTUNE)
```


```python
for x, y in int_train_ds.take(1):
    print(x.shape)
```


## Train an LSTM, for comparision

Do we really need these Transformers, anyway? Let's train a good old LSTM to form our baseline.


```python
lstm_model = keras.Sequential([
    keras.Input(shape=(None, max_length), dtype="int64"),
    keras.layers.Lambda(lambda x: tf.one_hot(x, depth=max_tokens)),
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation="sigmoid")
])

lstm_model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

lstm_model.summary()
```


Train the model.

Let's add a callback to save the best model, and then load it again before we measure the accuracy.


```python
callbacks = [
    keras.callbacks.ModelCheckpoint(
        "one_hot_bidir_lstm.keras",
        save_best_only=True
    )
]

lstm_model.fit(
    int_train_ds,
    validation_data=int_val_ds,
    epochs=10,
    callbacks=callbacks
)

lstm_model = keras.models.load_model("one_hot_bidir_lstm.keras")

print(f"Test acc: {lstm_model.evaluate(int_test_ds)[1]:.3f}")
```


## Define our Transformer model

For our encoder model we need a couple of different layers:

- `keras.layers.MultiHeadAttention`: The critical part, that adds the attention mechanism in parallel "heads".
- `keras.layers.Embedding`: Embeddings are great, so we will put our attention layer on top of an embedding layer.
- `keras.layers.LayerNormalization`: A normalisation layer that will improve the training.
- `keras.layers.Dense`: The classic dense layer, which will need to process the output features from the attention layers. Technically, we will say that the `Dense` layers compute a _projection_ of the features.


To make it all work in an efficient manner, we subclass the abstract `layer.Layer`.


```python
class TransformerEncoder(keras.layers.Layer):

    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        """
        Initalise our encoder
        """

        super().__init__(**kwargs)
        self.embed_dim = embed_dim  # Embedding dimensions
        self.dense_dim = dense_dim  # Dimensions of the following Dense layer
        self.num_heads = num_heads  # Number of attention heads

        # The important bit: The MultiHeadAttention layer
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        # Our projection part: Two Dense layers
        self.dense_proj = keras.Sequential(
            [keras.layers.Dense(dense_dim, activation="relu"),
             keras.layers.Dense(embed_dim),]
        )

        # Normalisation layers, one for each Dense layer.
        self.layernorm_1 = keras.layers.LayerNormalization()
        self.layernorm_2 = keras.layers.LayerNormalization()


    def call(self, inputs, mask=None):
        """
        The forward computations
        """

        # Apply a mask to ignore padded inputs (if any).
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        # Compute attention
        attention_output = self.attention(
            inputs, inputs, attention_mask=mask)

        # Compute the rest
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)

        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        """
        To save the model, we add a config method.
        """
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config
```


Cool. Let's instantiate the model:


```python
vocab_size = 20000
embed_dim = 256
num_heads = 2
dense_dim = 32

first_encoder_model = keras.Sequential([
    keras.Input(shape=(max_length,), dtype="int64"),
    keras.layers.Embedding(vocab_size, embed_dim),
    TransformerEncoder(embed_dim, dense_dim, num_heads),
    keras.layers.GlobalMaxPooling1D(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation="sigmoid")
])

first_encoder_model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

first_encoder_model.summary()
```


Train the encoder!


```python
callbacks = [
    keras.callbacks.ModelCheckpoint(
        "transformer_encoder.keras",
        save_best_only=True
    )
]

first_encoder_model.fit(
    int_train_ds,
    validation_data=int_val_ds,
    epochs=20, callbacks=callbacks
)

first_encoder_model = keras.models.load_model(
    "transformer_encoder.keras",
    custom_objects={"TransformerEncoder": TransformerEncoder})

print(f"Test acc: {first_encoder_model.evaluate(int_test_ds)[1]:.3f}")
```


Hmm. Maybe not entirely great yet?

We are missing a vital piece: So far, we are not really treating the inputs as a sequence, but rather just a set.

We need a mechanism for adding in the positions of the words in the input text. This mechanism is the _positional encoding_ scheme.


## Positional embeddings

Let's again create a custom layer, this time doing both the job of creating embeddings, and applying the positional encoding.


```python
class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)

        # Embeddings for the input tokens
        self.token_embeddings = keras.layers.Embedding(
            input_dim=input_dim, output_dim=output_dim
        )
        # Positional encoding. Notice the input dimensiond is the sequence length.
        self.position_embeddings = keras.layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

        # For computing the padding mask.
        self.not_equal = keras.layers.Lambda(lambda x: tf.math.not_equal(x, 0))

    def call(self, inputs):
        length = tf.shape(inputs)[-1]

        # Encode position just as an integer
        positions = tf.range(start=0, limit=length, delta=1)
        # Embed the positions.
        embedded_positions = self.position_embeddings(positions)

        # Embed tokens
        embedded_tokens = self.token_embeddings(inputs)

        # Sum the two
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return self.not_equal(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config
```


## Build the final transformer encoder

It's time to build and train the complete encoder. This will be almost identical to the previous one, except that we replace the single `Embedding` layer with our new, custom `PositionalEmbedding` layer.


```python
sequence_length = 600

second_encoder_model = keras.Sequential([
    keras.Input(shape=(max_length,), dtype="int64"),
    PositionalEmbedding(sequence_length, vocab_size, embed_dim),
    TransformerEncoder(embed_dim, dense_dim, num_heads),
    keras.layers.GlobalMaxPooling1D(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation="sigmoid")
])

second_encoder_model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
second_encoder_model.summary()
```


Train it!


```python
callbacks = [
    keras.callbacks.ModelCheckpoint(
        "full_transformer_encoder.keras",
        save_best_only=True
    )
]

second_encoder_model.fit(int_train_ds, validation_data=int_val_ds, epochs=20, callbacks=callbacks)

second_encoder_model = keras.models.load_model(
    "full_transformer_encoder.keras",
    custom_objects={
        "TransformerEncoder": TransformerEncoder,
        "PositionalEmbedding": PositionalEmbedding
    }
)

print(f"Test acc: {second_encoder_model.evaluate(int_test_ds)[1]:.3f}")
```


### <span style="color: red;">Exercise:<span>

With the `TransformerEncoder` layer in place, let's go ahead with our deep learning approach and stack several of them. The OG attention paper used 6 transformer blocks in their encoder, but maybe our performance on this dataset maxes out at 2 or 3? Try it out!

