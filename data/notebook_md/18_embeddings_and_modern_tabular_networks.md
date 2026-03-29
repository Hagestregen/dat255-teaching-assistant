<!-- source: 18_embeddings_and_modern_tabular_networks.ipynb -->

## Embeddings, and more modern networks for tabular data

One-hot encoding of categorical data is nice and effective, but let's try the embedding trick. We can run a standard dense network on top of the embeddings, but we also want to give it a go with _transformers_, in which case the embeddings are necessary.

We will split our feature is two types: numerical, which are normalised, and categorical, which are converted into embeddings.

Implementing the [TabTransformer](https://arxiv.org/abs/2012.06678) will be a difficult exercise, but mostly difficult in terms of sending the different features to the right parts of the network. Composing the network itself, using Keras components, is still quite convenient.


For out data, we will try to classify bank customers with good or bad credit risk. The dataset is described at the UCI ML dataset [repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).

Download and unzip:


```python
!wget https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip
!unzip -u statlog+german+credit+data.zip
```


Imports


```python
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
```


## Define feature types

The ones that are `"Categorical"` will go into the embeddings.


```python
feature_types = {
    "Existing checking account": "Categorical",
    "Duration": "Integer",
    "Credit history": "Categorical",
    "Purpose":"Categorical",
    "Credit amount": "Integer",
    "Savings account": "Categorical",
    "Employment since": "Categorical",
    "Installment rate": "Integer",
    "Personal status": "Categorical",
    "Other debtors": "Categorical",
    "Present residence since": "Integer",
    "Property": "Categorical",
    "Age": "Integer",
    "Other installment plans": "Categorical",
    "Housing": "Categorical",
    "Existing credits": "Integer",
    "Occupation": "Categorical",
    "Maintenance": "Integer",
    "Telephone": "Categorical",
    "Foreign worker": "Categorical",
}
```


Read the CSV file, for instance using Pandas


```python
import pandas as pd
dataframe = pd.read_csv("german.data", header=None, sep='\s+', names=list(feature_types.keys())+["Target"], index_col=False)
```


```python
dataframe.head()
```


## Prepare the data

We see that all categorical features are in fact strings with some weird string encoding. We need numbers to work with, so let's convert all to ordinal, integer encodings. We can use scikit-learn's `OrdinalEncoder`for this.

We do **not** convert to one-hot encodings in this case, because we will proceed with making embeddings.


```python
# Get the list of which features are categorical
categorical_features = [fname for fname, ftype in feature_types.items() if ftype == "Categorical"]

# Apply ordinal encoding for these
dataframe[categorical_features] = OrdinalEncoder().fit_transform(dataframe[categorical_features])
```


Let's see if it looks right:


```python
dataframe.head()
```


Great. Only one strange thing -- the targets are 1 and 2, not 0 and 1. Fix this by subtracting one.


```python
labels = dataframe.pop("Target")
labels = labels - 1
```


Create a train-test split


```python
X_train, X_test, y_train, y_test = train_test_split(dataframe, labels, test_size=0.2)
```


Create TensorFlow datasets, and batch them:


```python
train_ds = tf.data.Dataset.from_tensor_slices((dict(X_train), y_train))
test_ds = tf.data.Dataset.from_tensor_slices((dict(X_test), y_test))

batch_size = 32
train_ds = train_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)
```


## Define the model

We start by defining a function to prepare the inputs to our model -- just adding `Input` layers with the correct data type.


```python
def prepare_inputs():

    inputs = {}

    for feature_name, feature_type in feature_types.items():

        # Choose data type
        if feature_type in ["Binary", "Categorical"]:
            dtype = "int32"
        else:
            dtype = "float32"

        input_layer = keras.layers.Input(name=feature_name, shape=(1,), dtype=dtype)

        inputs[feature_name] = input_layer

    return inputs
```


Now for the important part:

Do normalisation of numerical features, and create embeddings for the categorical features.


```python
def process_inputs(inputs, embedding_dims):

    processed_inputs = {}

    for feature_name, input_layer in inputs.items():

        # Binary features: leave as they are
        if feature_types[feature_name] == "Binary":
            processed_inputs[feature_name] = input_layer

        # Numeric features: Apply normalisation
        elif feature_types[feature_name] == "Integer":

            norm_layer = keras.layers.Normalization(axis=None)
            norm_layer.adapt(X_train[feature_name].to_numpy())

            processed_inputs[feature_name] = norm_layer(input_layer)


        # Categorical features: Create embeddings
        elif feature_types[feature_name] == "Categorical":

            # Check how many categories we have
            num_categories = len(np.unique(X_train[feature_name]))

            # Add the embedding layer
            embedding_layer = keras.layers.Embedding(
                input_dim=num_categories,
                output_dim=embedding_dims
            )

            processed_inputs[feature_name] = keras.layers.Flatten()(embedding_layer(input_layer))

    return processed_inputs
```


For our first model -- Create a simple dense network.

At this point we have to select the dimensions for the embedding layers, which something that have to be optimised by testing.


```python
def create_simple_model():

    embedding_size = 16

    inputs = prepare_inputs()
    processed_inputs = process_inputs(inputs, embedding_size)
    all_inputs = keras.layers.concatenate(list(processed_inputs.values()))

    x = keras.layers.Dense(64, activation="relu", name="dense_1")(all_inputs)
    out = keras.layers.Dense(1, activation="sigmoid", name="dense_2")(x)

    model = keras.Model(inputs=inputs, outputs=out)

    return model
```


Instantiate the model, compile it, and have a look at the structure:


```python
simple_model = create_simple_model()
```


```python
simple_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```


```python
keras.utils.plot_model(simple_model, show_shapes=True)
```


Train it!


```python
simple_model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10,
    verbose=1
)
```


## Implement the TabTransformer

Let's take a stab at the [TabTransformer](https://arxiv.org/pdf/2012.06678), which looks like this:

<img src="https://raw.githubusercontent.com/keras-team/keras-io/master/examples/structured_data/img/tabtransformer/tabtransformer.png" width="450"/>

We see that the categorical features goes into embeddings, which we have done already, while the numerical features are normalised, which we also did. The remaing part is to add the layers of the network and connect the parts.


### <span style="color: red; font-weight: bold;">Exercise (difficult):<span>

Complete the TabTransformer model.

Some hints:
- You will have to collect the embedded features and the numerical features separately. In the simple model above we collected everything together like using `all_inputs = keras.layers.concatenate(list(processed_inputs.values()))`, so this has to be split into two.
- The basic structure of the transformer block is given below.
- You will need to use `keras.layers.concatenate` to merge the outputs from the transformer block with the numerical features.
- The network should end with `layers.Dense(units=1, activation="sigmoid")`.

Good luck!

For more hints, you can look at the relevant Keras [example](https://keras.io/examples/structured_data/tabtransformer/), which this notebook is based on. We use different data, but the idea is the same. The example does write the code in a more convoluted way, but most parts is similar to here.


```python
def transformer_block(categorical_features):

    # Self-attention: Call it two inputs, which are the same.
    attention_output = keras.layers.MultiHeadAttention(
        num_heads=2,
        key_dim=embedding_size,
        dropout=0.2
    )(categorical_features, categorical_features)

    # Skip connection 1
    x = keras.layers.Add()([attention_output, categorical_features])

    # Layer normalization 1
    x = keras.layers.LayerNormalization()(x)

    # Feedforward
    feedforward_output = keras.layers.Dense(64, activation="relu")(x)
    feedforward_output = keras.layers.Dropout(0.2)(feedforward_output)

    # Skip connection 2.
    x = keras.layers.Add()([feedforward_output, x])

    # Layer normalization 2.
    output = keras.layers.LayerNormalization()(x)

    return output
```


```python
def create_tabtransformer():

    pass
```


Train your model:


```python
tabtransformer.fit(
    train_ds,
    ...
)
```

