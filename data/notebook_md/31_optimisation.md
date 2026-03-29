<!-- source: 31_optimisation.ipynb -->

# Model size optimisation with LiteRT

In this notebook we will try to minimise our model size, in terms of both memory use and stored file size, using what is called _post-training quantisation_.

This means first training a model using full `float32` precision for the weights, and then converting them to 8-bit integers.

For all the details and more options for quantisation, have a look at the LiteRT documentation:
https://ai.google.dev/edge/litert/models/post_training_quantization.


Imports


```python
import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)
import tensorflow as tf
import keras
import numpy as np
import pathlib
```


## Train a model

Let's train a simple MNIST model to serve as our example.


```python
# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model architecture
model = keras.Sequential([
  keras.layers.InputLayer(input_shape=(28, 28)),
  keras.layers.Reshape(target_shape=(28, 28, 1)),
  keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation=tf.nn.relu),
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.Flatten(),
  keras.layers.Dense(10)
])

# Train the digit classification model
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
  train_images,
  train_labels,
  epochs=3,
  validation_data=(test_images, test_labels)
)
```


## Convert the model to LiteRT format

Converting to the optimised LiteRT format is quite simple -- we need just to call a `converter`-


```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```


Save it the converted model, without applying quantisation.


```python
tflite_models_dir = pathlib.Path("mnist_tflite_models/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
```


```python
tflite_model_file = tflite_models_dir/"mnist_model.tflite"
tflite_model_file.write_bytes(tflite_model)
```


## Quantise the model

Now, let's quantise all the parameters, using the `DEFAULT` strategy. This is a again just a matter of calling the `converter`, but we first add the optimisation strategy.
Quantise it


```python
# Set optimisation strategy
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert and save
tflite_quant_model = converter.convert()
tflite_model_quant_file = tflite_models_dir/"mnist_model_quant.tflite"
tflite_model_quant_file.write_bytes(tflite_quant_model)
```


Compare the file sizes:


```python
! du -h {tflite_models_dir}/*
```


That's almost a 1/4 reduction in size -- not bad for very little work.

Of course, we should check that we are not loosing prediction performance. Time to run the models.


## Run the optimised models


Running LiteRT models is a little differnt than a regular Keras one. In particular, we need an `Interpreter` to interface the model with its inputs and outputs.


```python
# The original model
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()
```


```python
# The quantised model
interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
interpreter_quant.allocate_tensors()
```


Load a test image, and predict:


```python
test_image = np.expand_dims(test_images[0], axis=0).astype(np.float32)

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

interpreter.set_tensor(input_index, test_image)
interpreter.invoke()
predictions = interpreter.get_tensor(output_index)
```


```python
import matplotlib.pylab as plt

plt.imshow(test_images[0])
template = "True:{true}, predicted:{predict}"
_ = plt.title(template.format(true= str(test_labels[0]),
                              predict=str(np.argmax(predictions[0]))))
plt.grid(False)
```


### Compare accuracies

Run over all the test images, and see how the quantised model compares to the non-quantised one.

First define a function to compute test accuracy


```python
# A helper function to evaluate the LiteRT model using "test" dataset.
def evaluate_model(interpreter):
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]

  # Run predictions on every image in the "test" dataset.
  prediction_digits = []
  for test_image in test_images:
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_image)

    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    output = interpreter.tensor(output_index)
    digit = np.argmax(output()[0])
    prediction_digits.append(digit)

  # Compare prediction results with ground truth labels to calculate accuracy.
  accurate_count = 0
  for index in range(len(prediction_digits)):
    if prediction_digits[index] == test_labels[index]:
      accurate_count += 1
  accuracy = accurate_count * 1.0 / len(prediction_digits)

  return accuracy
```


The run it:


```python
print(evaluate_model(interpreter))
print(evaluate_model(interpreter_quant))
```

