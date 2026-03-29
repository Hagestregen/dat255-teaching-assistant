<!-- source: 06-advanced-computer-vision.html -->
<!-- index-title: 6: Tuesday -->

# DAT255 – DAT255: Deep learning engineering
# DAT255: Deep learning engineering

Lecture 6 – Augmentation and advanced computer vision

sma@hvl.no

---

## Improving generalisation

**Realisation:**

- We don’t have infinite training data
- Training data don’t cover the space of all realistic examples

**Mitigation:**

- Add artificially modified duplicates of the training data (while preserving information)

 This is called **augmentation**

---

## Augmentation

The benefit of augmentation greatly exceeds the effort involved

Always recommended to use for computer vision.

---

## More advanced network configurations

Going beyond the `Sequential` model

---

## The Keras functional API

Consider the equivalent ways of defining a network:

The Sequential API:

```
from keras import layers
model = keras.Sequential([
  layers.Input(shape=input_shape),
  layers.Conv2D(64, 3, activation="relu"),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, activation="relu"),
  layers.Flatten(),
  layers.Dropout(0.5),
  layers.Dense(
    num_classes, activation="softmax"
  ),
])
```

Layers are stacked in a list

The Functional API

```
inputs = layers.Input(shape=input_shape)
x = layers.Conv2D(64, 3, activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, activation="relu")(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(
  num_classes, activation="softmax"
)(x)

model = keras.Model(
  inputs=inputs,
  outputs=outputs
)
```

Layers are used as *functions* taking the previous layer as input

Lastly we need to specify the input and output in a `Model`

---

## *Example:* Bird classifier

In the functional API we can easily have two input sources:

```
input1 = keras.layers.Input(shape=(128,128,3))

x1 = keras.layers.Conv2D(64, 3, activation='relu')(input1)
x1 = keras.layers.MaxPooling2D(2)(x1)
x1 = keras.layers.Conv2D(64, 3, activation='relu')(x1)
x1 = keras.layers.MaxPooling2D(2)(x1)
x1 = keras.layers.Flatten()(x1)

input2 = keras.layers.Input(shape=(10,))

x2 = keras.layers.Dense(32, activation='relu')(input2)
x2 = keras.layers.Dense(32, activation='relu')(x2)

concat = keras.layers.Concatenate()([x1, x2])

x = keras.layers.Dense(64, activation='relu')(concat)
out = keras.layers.Dense(3, activation='softmax')(x)

model = keras.Model(
  inputs=[input1, input2],
  outputs=out
)
```

 :::{.column width="40%"}

---

## Non-sequential networks

Can have networks with

- Multiple inputs
- Multiple outputs
- Arbitrary layer connections
- Networks-inside-the-network
- Loops (*will come to this later*)

Let’s look at some noteworthy architectures for computer vision.

---

## Residual connections

Implementing this branching and recombination can be done with Keras’ functional API:

```
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, activation="relu")(inputs)
# Set aside the residual
residual = x
# Conv block
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(2, padding="same")(x)
# Conv layer with strides=2, so that the shapes match
residual = layers.Conv2D(64, 1, strides=2)(residual)
# Add the block output with the residual
x = layers.add([x, residual])
```

If we need to this multiple times, we can write it in a

- `for` loop
- a function
- or as a subclass of `keras.layers.Layer`

---

## Residual networks

…which enables us to train networks with 100+ layers

He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition.* Proceedings of the IEEE conference on computer vision and pattern recognition.

---

## Densely connected convolutional networks

How about adding skip connections (almost) everywhere?

Enter the *DenseNet*:

Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). *Densely connected convolutional networks.* Proceedings of the IEEE conference on computer vision and pattern recognition.

---

## Inception networks

We can also add width in addition to depth:

The *inception* module features parallel convolutional layers with different kernel size

Output is concatenated and passed on to the next module

---

## Xception networks

The *Xception* (“extreme inception”) architecture relies on *depthwise separable* convolution layers

These layers are available as `keras.layers.SeparableConv2D` and can be used just like the regular `Conv2D`, often with increased performance

---

## Beyond CNNs: *Vision transformers*

Can also use methods from LLMs on image input * (get back to this in Ch. 15):

However, pure ConvNets are still hard to beat **

* Dosovitskiy, A. (2020). *An image is worth 16x16 words: Transformers for image recognition at scale.* arXiv:2010.11929
 ** Liu, Z., et al. (2022). *A convnet for the 2020s.* arxiv:2201.03545

---

## `keras.applications`

The most popular computer vision architectures are available as pre-trained models in `keras.applications`.

These are excellent starting points for

- feature extraction
- fine-tuning
- transfer learning

***Note***: The different models typically require specific preprocessing:

If you want to use

```
keras.applications.Xception()
```

you should process the input images with

```
keras.applications.xception.preprocess_input()
```

---

*[Interactive slide: Keras applications]*

---

## Feature extraction

---

## Fine-tuning

Keep general patterns learned, but make them more specific to a new task:

- Add custom network on top of a trained base network
- *Freeze* the base network
- Train the custom part
- *Unfreeze* the base network
- Jointly train the entire model

 ## Ablation studies

---

## Other computer vision tasks (next week)

Convolutional nets are great for other things than just classification:

- Object detection: Localise (several) objects in an image

- Oriented bounding boxes: Localise and estimate orientation of objects

- Semantic segmentation: Classify each pixel onto an object

- Instance segmentation: Draw a detailed outline around abjects

- Pose estimation: Localise distinctive features or parts of an object

---

## This week {background-color="#F9FBE7"}

:::{style="padding-top: 100px"}
:::

- Technicalities about loading data
- Data transformations (augmentation)
- Modern convnets for computer vision
- Other computer vision tasks

## Data pipelines

:::{style="padding-top: 50px"}
:::

In a deep learning setting, we typically need to consider that

1. Data are a too big to fit in memory _([problem]{.color-red-dark})_
2. We have two separate compute units, the CPU and the GPU _([opportunity]{.color-blue-dark})_

Implications:

1. Data must be divided into **_batches_**
2. While the GPU is working on one batch, the CPU can prepare the next one.

## Sequential processing

:::{style="padding-top: 100px"}
:::

![](https://www.tensorflow.org/guide/images/data_performance/naive.svg)

:::{.absolute top="33%" left="0%"}
Open
:::
:::{.absolute top="39.5%" left="0%"}
Read
:::
:::{.absolute top="46%" left="0%"}
Train
:::
:::{.absolute top="52%" left="0%"}
Epoch
:::

## Prefetching

:::{style="padding-top: 100px"}
:::

![](https://www.tensorflow.org/guide/images/data_performance/prefetched.svg)

:::{.absolute top="33%" left="0%"}
Open
:::
:::{.absolute top="39.5%" left="0%"}
Read
:::
:::{.absolute top="46%" left="0%"}
Train
:::
:::{.absolute top="52%" left="0%"}
Epoch
:::

## Prefetching + interleaving file reads

:::{style="padding-top: 100px"}
:::

![](https://www.tensorflow.org/guide/images/data_performance/parallel_interleave.svg)

:::{.absolute top="33%" left="0%"}
Open
:::
:::{.absolute top="39.5%" left="0%"}
Read
:::
:::{.absolute top="46%" left="0%"}
Train
:::
:::{.absolute top="52%" left="0%"}
Epoch
:::

## Prefetching + interleaving + parallel preprocessing

:::{style="padding-top: 100px"}
:::

![](https://www.tensorflow.org/guide/images/data_performance/parallel_map.svg)

:::{.absolute top="33%" left="0%"}
Open
:::
:::{.absolute top="40.5%" left="0%"}
Read
:::
:::{.absolute top="47%" left="0%"}
Map
:::
:::{.absolute top="53.5%" left="0%"}
Train
:::
:::{.absolute top="60%" left="0%"}
Epoch
:::

## TensorFlow `Dataset`

:::{style="padding-top: 100px"}
:::

Get all these features with [minimal effort]{.color-green.dark} though `tf.data.Dataset`

:::{style="padding-top: 40px"}
:::

:::{.fragment}
[_Effort_]{.color-red-dark}: Getting the data into a `Dataset`.

[_Reduced effort_]{.color-green-dark}: Keras convenience functions for "standard" data.
:::

## TensorFlow `Dataset`

As an example, create a `Dataset` from a list: (_although we never do this in practice_)

```{.python code-line-numbers="1|1-3|4,5|4-8"}
>>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
>>> dataset
<_TensorSliceDataset element_spec=TensorSpec(shape=(), dtype=tf.int32, name=None)>
>>> for element in dataset:
>>>   print(element)
tf.Tensor(1, shape=(), dtype=int32)
tf.Tensor(2, shape=(), dtype=int32)
tf.Tensor(3, shape=(), dtype=int32)
```

Apply some transformation:
```{.python code-line-numbers="1,2|4|"}
>>> def square(x):
>>>    return x * x

>>> new_dataset = dataset.map(square)

>>> for element in new_dataset:
>>>    print(element)
tf.Tensor(1, shape=(), dtype=int32)
tf.Tensor(4, shape=(), dtype=int32)
tf.Tensor(9, shape=(), dtype=int32)
```

**Important:** The `Dataset` methods do not modify data in-place, but always returns a new `Dataset`.

## TensorFlow `Dataset`

Assuming we already have a `Dataset`, set up the parallel processing chain:

:::{style="padding-top: 20px"}
:::

```{.python code-line-numbers="1|2|3|4|"}
shuffled_ds = dataset.shuffle()  # optional
preprocessed_ds = shuffled_ds.map(preprocessing_fn, num_parallel_calls=10)
batched_ds = preprocessed_ds.batch(batch_size)
prefeched_ds = batched_ds.prefetch(buffer_size)
```

or as a one-liner:

:::{style="padding-top: 20px"}
:::

```{.python}
ds = dataset.shuffle().map(...).batch(...).prefetch(...)
```

:::{.fragment}
In most cases we can set `buffer_size`, `num_parallel_calls`, etc to [`tf.data.AUTOTUNE`](https://www.tensorflow.org/guide/data_performance) and have TensorFlow figure out the best setting for us.

:::{style="padding-top: 20px"}
:::
```{.python}
dataset.prefetch(tf.data.AUTOTUNE)
```
:::

## Getting data into a `Dataset`

Can be tricky since `Dataset`s are maximally generic, but Keras to the rescue for the most common data types:

[**_I have_**]{.color-indigo}

- Images: `keras.utils.image_dataset_from_directory`
- Time series: `keras.utils.timeseries_dataset_from_array`
- Text: `keras.utils.text_dataset_from_directory`
- Audio: `keras.utils.audio_dataset_from_directory`

[**_While if I have_**]{.color-indigo}

- CSV files: Read the TensorFlow [tutorial](https://www.tensorflow.org/tutorials/load_data/csv)
- Something else: Code it yourself

## Train a model

:::{style="padding-top: 30px"}
:::

`Dataset`s are input to `.fit()` just as usual:

:::{style="padding-top: 20px"}
:::
```{.python}
model.fit(
    train_dataset,
    validation_data=val_dataset,
    ...
)
```
and Keras figures out the rest.

See this week's notebook and consult the [documentation](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) for more info on `Dataset`s.
