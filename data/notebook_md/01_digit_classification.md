<!-- source: 01_digit_classification.ipynb -->

# Tutorial: Digit classification using convolutional networks

In this notebook we will get familiar with the basics of Keras by trying to classify images of handwritten digits, which is known as the MNIST dataset. The images contain single digits (so numbers 0 to 9), meaning we have to do multiclass classification. 

This is a relatively simple task for us to do today, but was among the coolest problems you could work on in 1998. We will try out two different models -- first a plain feed-forward neural network, then one based on convolutions. 

This notebook is based on one of the Keras examples: https://keras.io/examples/vision/mnist_convnet/ Here you will find a lot on interesting examples of computer vision models -- many of them rather advanced. This one is, however, a nice starting point.


## Setup
Import the libraries we need.


```python
import numpy as np
import keras
import matplotlib.pyplot as plt
```

**Output:**

```
2026-01-23 15:04:25.762786: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2026-01-23 15:04:25.783081: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8473] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2026-01-23 15:04:25.789641: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1471] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2026-01-23 15:04:25.805291: I tensorflow/core/platform/cpu_feature_guard.cc:211] This TensorFlow binary is optimized to use available CPU instructions in performance
… [output truncated]
```


## Load and prepare the data


```python
# Ten classes (numbers 0 to 9)
num_classes = 10

# The images are 28x28 pixels, and have one channel (grayscale). 
# For color images, the last number here would be 3 (red/green/blue channel).
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)
print("X_train shape:", X_train.shape)
print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```

**Output:**

```
X_train shape: (60000, 28, 28, 1)
60000 train samples
10000 test samples
```


Look at some of the data.


```python
n_rows = 5
n_cols = 5
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
        plt.axis('off')
plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()
```

**Output:**

_[image output — see notebook]_


## Model number 1: A simple feed-forward network

This "multi-layer perceptron" tries to recognise patterns based on the value of each individual pixel, at its fixed position in the image.

First, we define the model and print it:


```python
ff_network = keras.Sequential(
    [
        keras.layers.Input(shape=input_shape),
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ]
)

ff_network.summary()
```

**Output:**

```
/usr/local/lib/python3.12/dist-packages/keras/src/layers/reshaping/flatten.py:37: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(**kwargs)
2026-01-23 15:04:30.793659: I tensorflow/core/common_runtime/gpu/gpu_device.cc:2021] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 22233 MB memory:  -> device: 0, name: NVIDIA RTX A5000, pci bus id: 0000:84:00.0, compute capability: 8.6
2026-01-23 15:04:30.794096: I tensorflow/core/common_runtime/gpu/gpu_device.cc:2021] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 9781 MB memory:  -> device: 1, name: NVIDIA GeForce RTX 2080 Ti, pci bus id: 0000:04:00.0, compute 
… [output truncated]
```
_[HTML output — see notebook]_
_[HTML output — see notebook]_
_[HTML output — see notebook]_
_[HTML output — see notebook]_
_[HTML output — see notebook]_


Notice that we used the _sequential model API_: We made a list of layers and gave it as input to [`keras.Sequential`](https://keras.io/guides/sequential_model/). This is the most convenient way to create a model where each layer connects directly to the next. In case we need a more advanced layout of the model, we will have to use the _functional model API_, which we will look at later.


Next, we need to specify the loss function, the optimisation algorithm to use, and any metrics we want to measure during training. Since the classes (0-9) are evenly distributed, we choose to measure the accuracy.
"Compiling" the model configures it according to the given specification.


```python
ff_network.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
```


Now we are ready to train the model:


```python
batch_size = 128    # How many images to load in a single batch
epochs = 10         # How many times to iterate over the full dataset

ff_network.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
```

**Output:**

```
Epoch 1/10
```
```
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1769177072.559069   28486 service.cc:146] XLA service 0x7fe64000a0c0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
I0000 00:00:1769177072.559118   28486 service.cc:154]   StreamExecutor device (0): NVIDIA RTX A5000, Compute Capability 8.6
I0000 00:00:1769177072.559122   28486 service.cc:154]   StreamExecutor device (1): NVIDIA GeForce RTX 2080 Ti, Compute Capability 7.5
2026-01-23 15:04:32.625248: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:268] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
2026-01-23 15:04:32.829581: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:531] Loaded cuD
… [output truncated]
```
```
[1m171/422[0m [32m━━━━━━━━[0m[37m━━━━━━━━━━━━[0m [1m0s[0m 888us/step - accuracy: 0.6064 - loss: 1.3850
```
```
I0000 00:00:1769177073.526657   28486 device_compiler.h:188] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
```
```
[1m409/422[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 865us/step - accuracy: 0.7343 - loss: 0.9581
```
```
'+ptx85' is not a recognized feature for this target (ignoring feature)
'+ptx85' is not a recognized feature for this target (ignoring feature)
'+ptx85' is not a recognized feature for this target (ignoring feature)
```
```
[1m422/422[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.7381 - loss: 0.9448
```
```
'+ptx85' is not a recognized feature for this target (ignoring feature)
'+ptx85' is not a recognized feature for this target (ignoring feature)
'+ptx85' is not a recognized feature for this target (ignoring feature)
```
```
[1m422/422[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 4ms/step - accuracy: 0.7384 - loss: 0.9438 - val_accuracy: 0.9420 - val_loss: 0.2117
Epoch 2/10
[1m 59/422[0m [32m━━[0m[37m━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 871us/step - accuracy: 0.9309 - loss: 0.2532
```
```
'+ptx85' is not a recognized feature for this target (ignoring feature)
'+ptx85' is not a recognized feature for this target (ignoring feature)
'+ptx85' is not a recognized feature for this target (ignoring feature)
```
```
[1m422/422[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9308 - loss: 0.2436 - val_accuracy: 0.9515 - val_loss: 0.1677
Epoch 3/10
[1m422/422[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1000us/step - accuracy: 0.9444 - loss: 0.1970 - val_accuracy: 0.9555 - val_loss: 0.1509
Epoch 4/10
[1m422/422[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9515 - loss: 0.1699 - val_accuracy: 0.9625 - val_loss: 0.1334
Epoch 5/10
[1m422/422[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9571 - loss: 0.1462 - val_accuracy: 0.9638 - val_loss: 0.1201
Epoch 6/10
[1m422/422[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9611 - loss: 0.1343 - val_accuracy: 0.9638 - val_loss: 0.124
… [output truncated]
```
```
<keras.src.callbacks.history.History at 0x7fe8d8a20c50>
```


Let's evaluate the network on the test data:


```python
score = ff_network.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
```

**Output:**

```
'+ptx85' is not a recognized feature for this target (ignoring feature)
'+ptx85' is not a recognized feature for this target (ignoring feature)
'+ptx85' is not a recognized feature for this target (ignoring feature)
```
```
Test loss: 0.1250554770231247
Test accuracy: 0.9632999897003174
```
```
'+ptx85' is not a recognized feature for this target (ignoring feature)
'+ptx85' is not a recognized feature for this target (ignoring feature)
'+ptx85' is not a recognized feature for this target (ignoring feature)
```


## A better network: The convolutional neural network (_convnet_)

Now let's construct a model based on convolutional and pooling layers.


```python
convnet = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

convnet.summary()
```

**Output:**

_[HTML output — see notebook]_
_[HTML output — see notebook]_
_[HTML output — see notebook]_
_[HTML output — see notebook]_
_[HTML output — see notebook]_


Compile and train it on the same data as before:


```python
convnet.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

convnet.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
```

**Output:**

```
Epoch 1/10
```
```
'+ptx85' is not a recognized feature for this target (ignoring feature)
'+ptx85' is not a recognized feature for this target (ignoring feature)
'+ptx85' is not a recognized feature for this target (ignoring feature)
```
```
[1m383/422[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 1ms/step - accuracy: 0.7142 - loss: 0.9357
```
```
'+ptx85' is not a recognized feature for this target (ignoring feature)
'+ptx85' is not a recognized feature for this target (ignoring feature)
'+ptx85' is not a recognized feature for this target (ignoring feature)
```
```
[1m422/422[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 0.7279 - loss: 0.8916
```
```
'+ptx85' is not a recognized feature for this target (ignoring feature)
'+ptx85' is not a recognized feature for this target (ignoring feature)
'+ptx85' is not a recognized feature for this target (ignoring feature)
```
```
[1m422/422[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 6ms/step - accuracy: 0.7282 - loss: 0.8906 - val_accuracy: 0.9717 - val_loss: 0.1040
Epoch 2/10
[1m  1/422[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m10s[0m 24ms/step - accuracy: 0.9766 - loss: 0.0992
```
```
'+ptx85' is not a recognized feature for this target (ignoring feature)
'+ptx85' is not a recognized feature for this target (ignoring feature)
'+ptx85' is not a recognized feature for this target (ignoring feature)
```
```
[1m422/422[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - accuracy: 0.9531 - loss: 0.1553 - val_accuracy: 0.9802 - val_loss: 0.0687
Epoch 3/10
[1m422/422[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - accuracy: 0.9671 - loss: 0.1068 - val_accuracy: 0.9832 - val_loss: 0.0614
Epoch 4/10
[1m422/422[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.9721 - loss: 0.0913 - val_accuracy: 0.9852 - val_loss: 0.0512
Epoch 5/10
[1m422/422[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.9757 - loss: 0.0809 - val_accuracy: 0.9868 - val_loss: 0.0489
Epoch 6/10
[1m422/422[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.9777 - loss: 0.0727 - val_accuracy: 0.9862 - val_loss: 0.0477
E
… [output truncated]
```
```
<keras.src.callbacks.history.History at 0x7fe8b02fac00>
```


And evaluate the results:


```python
score = convnet.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
```

**Output:**

```
'+ptx85' is not a recognized feature for this target (ignoring feature)
'+ptx85' is not a recognized feature for this target (ignoring feature)
'+ptx85' is not a recognized feature for this target (ignoring feature)
```
```
Test loss: 0.0336548313498497
Test accuracy: 0.9889000058174133
```
```
'+ptx85' is not a recognized feature for this target (ignoring feature)
'+ptx85' is not a recognized feature for this target (ignoring feature)
'+ptx85' is not a recognized feature for this target (ignoring feature)
```


Notice how we made a smaller model (17k vs 26k number of parameters), but still got better results.

