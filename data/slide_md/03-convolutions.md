<!-- source: 03-convolutions.html -->
<!-- index-title: 3: Monday -->

# DAT255 – DAT255: Deep learning engineering
# DAT255: Deep learning engineering

Lecture 3 – Computer vision and the concepts of layers

sma@hvl.no

---

*[Interactive slide: TensorFlow playground]*

---

## *Shallow* learning

\[
y = f(x_1, x_2, x_3, x_4, \dots, x_n | \theta)
\]

---

## *Deep* learning

\[
y = f_3(f_2(f_1(x_1, x_2 | \theta_1) | \theta_2) | \theta_3)
\]

---

*[Interactive slide: TensorFlow playground]*

---

## *Deep* learning

The point of deep learning is to sequentially **learn better feature representations**, and use these to solve a task.

*insufficient:*


 *good:*


 *better:*

data


 data


 data

\(\rightarrow\)


 \(\rightarrow\)


 \(\rightarrow\)

prediction


 representation


 representation

\(\rightarrow\)


 \(\rightarrow\)

prediction


 representation

\(\rightarrow\)

prediction

Since neural networks are *universal function approximators*, they can model arbitrarily complex relationships. The cost of doing so, is that we need a lot of data.

---

## The feed-forward neural network

For the demo we used the good ol’ fully-connected feed-forward network:

Each node computes an output by

\[
\small
y = f\left(b + \sum_{i=1}^{n} w_ix_i\right)
\]

where

- \(w_i\) is the weight of each incoming connection
- \(b\) is the bias term
- \(f\) is the activation function (more tomorrow)

Now we want to motivate new types of layers.

---

## Image classification

Let’s classify this image: *(see notebook 1)*

Try to treat every pixel as feature:

not 2

Two obvious problems:

- Not invariant under translation (move the image  different result)
- Not invariant under dilation (resize the image  different result)

---

## Enter the *convolution operation*

The foundation for modern computer vision *(plus lots of other things!)* is **convolution**:

an operation that takes in two functions and returns a new function

\[
\small
f \ast g \equiv \int_{-\infty}^{\infty} f(\tau) g(t+\tau) d\tau
\]

In practice, convolution is a way to **recognise and localise patterns** in data

Technical note:

In signal processing we would call \(\small\int_{-\infty}^{\infty} f(\tau) g(t+\tau) d\tau\) *cross-correlation* and the time-reversed version \(\small\int_{-\infty}^{\infty} f(\tau) g(t-\tau) d\tau\) would be convolution. In ML these got mixed up and we just talk about convolution.

---

## Discrete convolution

Convolution is a lot easier with discrete data such as images, because:

- the integral becomes a sum
- the first function is our image
- the second function is our ***kernel*** or ***filter***, which tries to find patterns in the image.

---

## Convolution over images

For RGB color images: Process each color channel, then sum

---

## The convolution *kernel* (or *filter*)

Convolution with predefined kernels is the core to digital image processing
 (but then we call it *filters*)

---

## More filters

Average: Blurring effect

*Sobel* filter: Edge detector

---

## Kernels for image recognition

Let’s try handcrafting some filters/kernels:

Need to refine
the approach :/

---

## Decomposition into simple patters

---

Talk is cheap.
Show me the code

---

## Keras layers

For our proposed solution we need three layer types:

- **Convolution layers** to extract image features `keras.layers.Conv2D`

- **Pooling layers** to downsample and aggregate the features `keras.layers.MaxPooling2D`

- **Fully-connected (*dense*) layers** to compute the final prediction `keras.layers.Dense`

---

## The `Conv2D` layer

```
keras.layers.Conv2D(
    filters,
    kernel_size,
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    groups=1,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
```

**Padding** around the image edges
 *(typical: 1 or 2)*

---

## Pooling layers

Two reasons for downsampling the image features:

- Learn a **spatial hierarchy** by widening the *receptive field*
- Reduce the total parameter count

Most common approach: Choose the maximum value from a window of pixels

---

## The `MaxPooling2D` layer

```
keras.layers.MaxPooling2D(
    pool_size=(2, 2),
    strides=None,
    padding="valid",
    data_format=None,
    name=None,
    **kwargs
)
```

`pool_size`: Size of pooling window
 *(typical: 2x2 or 3x3)*

`strides`: Step size
 *(typical: same as `pool_size`)*

`padding`: Padding around edges, `valid` or `same`

---

## The `Dense` layer

```
keras.layers.Dense(
    units,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    lora_rank=None,
    **kwargs
)
```

`units`: number of nodes in this layer

In the *last* dense layer, `units` must be equal to the number of classes (or otherwise number of desired outputs).

---

## My first convolutional network ✨

Configure the training objective and strategy:

```
convnet.compile(
  loss="categorical_crossentropy",
  optimizer="adam",
  metrics=["accuracy"]
)
```

(again, more details later + in the textbook)

Start training!

```
convnet.fit(
  X_train,
  y_train,
  batch_size=128,
  epochs=15,
  validation_split=0.1
)
```

---

## Decomposition into simple patters: Theory vs practice

Our test image:

From chapter 10, *Interpreting what ConvNets learn*

---

## Layer *activations*

Repeat for all filters in all layers:

**Layer 5 (last):**

---

*[Interactive slide: CNN explainer]*
