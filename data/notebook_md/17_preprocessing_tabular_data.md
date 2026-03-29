<!-- source: 17_preprocessing_tabular_data.ipynb -->

# Preprocessing tabular data with Keras

Keras offers various preprocessing functions for tabular data, which are implemented as layers and can be used as part of a model, just like the ordinary model layers we have used so far.

Some of them (like `Normalisation`) need to be adapted to data before we start training, an require a bit of extra work before we can get started. We'll do this extra work in this notebook.

The data we use contains diagnostic information related to coronary artery disease, and our goal is to predict the presence of disease. The details of te dataset are available at the UCI ML dataset [repository](https://archive.ics.uci.edu/dataset/45/heart+disease).

You can also find additional examples of tabular data preprocessing in the [Keras examples](https://keras.io/examples/structured_data/) section.

In this notebook there are no real exercises (apart from the last one), it is more to show how things can be done. But try to pay attention wehn running it so that you understand what is going on.


```python
import pandas as pd
import tensorflow as tf
import keras
```


We can try reading the CSV file using Pandas this time.


```python
file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
dataframe = pd.read_csv(file_url)
```


```python
dataframe.head()
```


Remove the target feature, and create a TensorFlow dataset:


```python
labels = dataframe.pop("target")
labels = tf.expand_dims(labels, -1)
ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
```


If we like, we can test the contents of our dataset object.


```python
for x, y in ds.take(1):
    print(x, y)
```


## Define feature types, and set up functions for data proprocessing

We want to treat numerical and categorical features differently, and in this case there are also two variants of categorical features -- those encodes as integers (ordinal), and those encoded as a string.

Let's call them `numerical`, `categorical_integer`, and `categorical_string`.


```python
feature_types = {
    "age":      "numerical",
    "sex":      "categorical_integer",
    "cp":       "categorical_integer",
    "trestbps": "numerical",
    "chol":     "numerical",
    "fbs":      "categorical_integer",
    "restecg":  "categorical_integer",
    "thalach":  "numerical",
    "exang":    "categorical_integer",
    "oldpeak":  "numerical",
    "slope":    "numerical",
    "ca":       "numerical",
    "thal":     "categorical_string"
}
```


For the actual treatment of the different features -- loop over all of them and declare the relevant preprocessing layer to apply.

We collect them all in a dict.


```python
processing_layers = {}


for feature_name, feature_type in feature_types.items():

    # In order to adapt the layers to data, we need a dataset that
    # contains *only* this feature. Create it here.
    feature_ds = ds.map(lambda x, y: x[feature_name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))


    if feature_type == "numerical":

        # Numerical data -> Normalise
        normaliser = keras.layers.Normalization()
        normaliser.adapt(feature_ds)

        processing_layers[feature_name] = normaliser

    if feature_type == "categorical_integer":

        # Ordinal data -> Create one-hot encoded features
        integer_lookup = keras.layers.IntegerLookup(output_mode="one_hot")
        integer_lookup.adapt(feature_ds)

        processing_layers[feature_name] = integer_lookup

    if feature_type == "categorical_string":

        # String data -> Create one-hot encoded features
        string_lookup = keras.layers.StringLookup(output_mode="one_hot")
        string_lookup.adapt(feature_ds)

        processing_layers[feature_name] = string_lookup


# Print just to see if it looks right
for name, layer in processing_layers.items():
    print(name, layer)
```


## Apply the pre-processing

Here we have two options:
1. Apply to the TensorFlow dataset, and create a new dataset containing preprocessed data
2. Add it as part of the Keras model

In this case we choose option 1, while in the next notebook we have a look at option 2.


A short wrapper function to apply our dict of layers


```python
def apply_preprocessing(features, target):

    for feature_name in features:

        layer = processing_layers[feature_name]
        features[feature_name] = layer(features[feature_name])

    return dict(features), target
```


Apply it to the dataset to create a new one:


```python
processed_ds = ds.map(apply_preprocessing)
```


Again we can hava a look to be sure.

Note that now the different features can have different shapes, because the one-hot encoding adds new columns.


```python
for x, y in processed_ds.take(1):
    for name in x:
        print(name, x[name])
    print("target:", y)
```


## Define the model

All Keras models start with an `Input` layer, but here we need to connect the different outputs to an `Input` layer with the correct shape.

Let's connect them by name and set the appropriate data type and shape.


```python
input_layers = {}
for feature_name, feature_type in feature_types.items():

    if feature_type == "numerical":
        input_layers[feature_name] = keras.layers.Input(name=feature_name, shape=(1,), dtype="float32")

    else:
        num_categories = len(processing_layers[feature_name].get_vocabulary())
        input_layers[feature_name] = keras.layers.Input(name=feature_name, shape=(num_categories,), dtype="int32")


for n, l in input_layers.items():
    print(n, l)
```


### Model definition as a class

Let's look at a more advanced way to define a model -- by subclassing the base `Layer` class.

Here we need three functions:
- `__init__`, containing the layers our model will contain,
- `call`, stating what happens when we run the model, and
- `build`, which doesn't have to contain anything, but must be defined.

This way of defining a model is the standard way for the other big deep learning library, called [PyTorch](https://pytorch.org/). So in case you happen to use that at some later point, you will recognise the structure.


```python
class MyModel(keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):

        # We could also have put the input and processing layers here,
        # if not applying them directly to the TF dataset.

        all_features = keras.layers.concatenate(list(inputs.values()))  # merge the different inputs
        x = self.dense1(all_features)   # apply the first Dense layer
        output = self.dense2(x)         # apply the second (classification) layer

        return output

    def build(self, input_shape):
        self.built = True
```


Create a `keras.Model()` instance


```python
def create_model():
    output = MyModel()(input_layers)
    model = keras.Model(input_layers, output)

    return model
```


```python
model = create_model()
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```


Plot the model, to see where the data goes:


```python
keras.utils.plot_model(model, show_shapes=True, rankdir="TD")
```


## Run the model

Run it on the processed dataset.

Now we haven't created a separate validation dataset, but this you can fix yourself :)


```python
model.fit(processed_ds, epochs=10)
```


### <span style="color: red; font-weight: bold;">Exercise:<span>

Study the class implementation of the network (`MyModel`) above. Try to improve it by adding
- A second hidden `Dense` layer
- Batch normalisation
- `Dropout` between the `Dense` layers.

