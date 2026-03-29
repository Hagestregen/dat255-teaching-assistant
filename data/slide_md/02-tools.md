<!-- source: 02-tools.html -->
<!-- index-title: 2: Tuesday -->

# DAT255 – DAT255: Deep learning engineering
# DAT255: Deep learning engineering

Lecture 2 – The tools of deep learning

sma@hvl.no

---

## Machine learning

---

## Classification

Classification threshold:

x < 0.5

x > 0.5

---

*[Interactive slide: TensorFlow playground]*

---

## What do we need from our tools

A good deep learning framework should provide

- A data structure for representing \(\small{}N\)-dimensional arrays (\(\small N \in \{0, 1, 2, \dots\}\)), aka *tensors*
- A way to compute derivatives of arbitrary functions
- A way to run computations on hardware accelerators, such as GPUs

There are many of these; luckily most have converged to a “NumPy-like” API.

Read about similarities and differences in Ch. 3 (optional)

---

## Frameworks

`python` is the de-facto language for deep learning

In case you need a refresher, look at e.g.

- *Kaggle Learn:* https://www.kaggle.com/learn/python
- *Google Edu:* https://developers.google.com/edu/python

Technically, we use python as a configuration language while the framework backends
are C++/CUDA (but we won’t touch this)

For deployment, there are hooks to Haskell / C# / Julia / Java / R / Ruby / Rust / Scala / Perl and others

*We assume you have been exposed to NumPy – although it’s no requirement*

---

## This week

**Environment setup:** Check our GitHub

**Intro to TensorFlow:**
We look at this today

---

## Computing resources

Many exercises in this course are compute intensive, and will benefit from a *hardware accelerator* (i.e. a GPU)

Some options:

- Your own computer (NVIDIA GPU or M-series Mac)
- Cloud services - Google Colab: T4 for free - Kaggle Notebooks: P100 for free - (and others)
- Research group hardware If you are connected to some research group at HVL/UiB

---

## Low-level TensorFlow

Most things work like NumPy, but with the benefit of GPU support and JIT compilation.

The core object is the `Tensor`, which is basically a multidimensional array.

NumPy

```
>>> import numpy as np
>>> x = np.array([[1,2,3], [4,5,6]], dtype=np.float32)
>>> print(x)
[[1. 2. 3.]
 [4. 5. 6.]]
```

TensorFlow

```
>>> import tensorflow as tf
>>> x = tf.constant([[1,2,3], [4,5,6]], dtype=tf.float32)
>>> print(x)
tf.Tensor(
[[1. 2. 3.]
 [4. 5. 6.]], shape=(2, 3), dtype=float32)
```

---

## Low-level TensorFlow: The `Tensor`

`Tensor`s are immutable, and are useful for storing constant values such as input data

For values that will be updated (such as model weights), use a `Variable`:

```
>>> y = tf.Variable([[1,2,3], [4,5,6]], dtype=tf.float32, name="My first variable")
>>> y[0,1].assign(50)
<tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32, numpy=
array([[ 1., 50.,  3.],
       [ 4.,  5.,  6.]], dtype=float32)>
```

---

## Low-level TensorFlow: Simple operations

Basic math is accessed through operators

```
>>> a = b + c       # Element-wise addition
>>> a = b * c       # Element-wise multiplication (Hadamard product)
>>> a = b @ c       # Matrix multiplication
```

while more complicated stuff is available as functions in `tf.math`

Notice in particular the `reduce_` functions, which look different from NumPy:

```
>>> x = tf.constant([[1,2,3], [4,5,6]]); print(x)
tf.Tensor(
[[1 2 3]
 [4 5 6]], shape=(2, 3), dtype=int32)

>>> tf.math.reduce_sum(x, axis=0)
<tf.Tensor: shape=(3,), dtype=int32, numpy=array([5, 7, 9], dtype=int32)>

>>> tf.math.reduce_sum(x, axis=1)
<tf.Tensor: shape=(2,), dtype=int32, numpy=array([ 6, 15], dtype=int32)>

>>> tf.math.reduce_sum(x, axis=None)
<tf.Tensor: shape=(), dtype=int32, numpy=21>
```

---

## Low-level TensorFlow: *Shapes*

*Broadcasting* works like in NumPy:

```
>>> x = tf.constant([1,2,3], dtype=tf.float32)
>>> x + 1
<tf.Tensor: shape=(3,), dtype=float32, numpy=array([2., 3., 4.], dtype=float32)>
```

Same for *shapes*:

---

## A typical shape: [128, 299, 299, 3]

Images are typically represented as `[height, width, channel]`

During model training we want to do *minibatch gradient descent*, and load a subset of the data at a time

This adds a fourth **batch** dimension: `[batch, height, width, channel]`

When processing single data points, we often need to add or remove it:

```
img = tf.expand_dims(img, 0)    # [24, 24, 3]    -> [1, 24, 24, 3]
img = tf.squeeze(img)           # [1, 24, 24, 3] -> [24, 24, 3]
```

---

## Run computations on a GPU

TensorFlow will automatically try to use the fastest compute device available

Tensor operations are typically faster when parallelised on a GPU

While this is automatic, it can also be forced:

```
with tf.device('CPU:0'):
  a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.Variable([[1.0, 2.0, 3.0]])

with tf.device('GPU:0'):
  k = a * b

print(k)
```

---

## Automatic differentiation

Let’s try to compute this derivative:

\[
\small
\frac{\mathrm{d}}{\mathrm{d}x} \Big|_{x=1} \; x^2 + 2x - 5
\]

\[
\small
= 2x + 2 \big|_{x=1} = 2 \cdot 1 + 2 = 4
\]

Turns out TensorFlow can do it *for us:*

```
def f(x):
  return x**2 + 2*x - 5

x = tf.Variable(1.0)

with tf.GradientTape() as tape:
  y = f(x)

d_dx = tape.gradient(y, x)
print(d_dx)
```

```
<tf.Tensor: shape=(), dtype=float32, numpy=4.0>
```

---

## Solving linear regression with autodiff

Let’s write the loss function in TensorFlow:

\[
\small
\begin{align}
\color{MediumVioletRed}{L}_{\mathrm{MSE}}
&= \frac{1}{N}\sum_{i=1}^{N} \left(\boldsymbol{\color{teal}{\theta}}^{\intercal} \color{DarkOrange}{\mathbf{x}}^{(i)} - \color{DarkBlue}{y}^{(i)}\right)^2
\end{align}
\]

```
def mse(y_pred, y_true):
  return tf.reduce_mean(
    tf.math.square(y_pred - y_true)
  )
```

Notice this is a **differentiable function**.

So we have

1. A differentiable function *(mean squared error)*
2. A tool that can compute derivatives *(TensorFlow)*

which means we are all set to do gradient descent.

---

## Gradient descent

Recalling that minimum loss (error) is where the derivative of the function is zero, let’s find the optimal parameters `theta` numerically:

```
theta = tf.Variable([0.21, 1.43]) # random values

for epoch in range(10):

  with tf.GradientTape() as tape:
    y_pred = X.dot(theta)
    loss = mse(y_pred, y_true)

  better_theta = theta - learning_rate * tape.gradient(loss, theta)

  theta.assign(better_theta)
```

This loop is at the core of all neural network training

(but with Keras we don’t write it out explicitly)

---

## Artificial neural networks

The base element of a neural network is the neuron, which looks a lot like multiple linear regression: \[
\small
a = b + \sum_{i=1}^{M} w_ix_i
\]

But we can solve *nonlinear* problems if we add a nonlinear transformation \(\small{}f\),

\[
\small
y = f(a)
\]

which we call an activation function.

Again we usually don’t bother writing this computation out explicitly, but rather interact with Keras *layers*.

This one is called `keras.layers.Dense`, but we’ll look at several types next week

 ## Number of bits

---

## Keras

The Keras framework contains all the high-level components we need to construct and train a neural network:

- `keras.layers`: Different types of layers and activation functions
- `keras.callbacks`: Monitor, modify or stop the training process
- `keras.optimizers`: Optimisation algorithms
- `keras.metrics`: Performance metrics
- `keras.losses`: Loss functions
- `keras.datasets`: Small datasets for testing
- `keras.applications`: Pre-trained networks for different tasks

---

## Referansegruppe
