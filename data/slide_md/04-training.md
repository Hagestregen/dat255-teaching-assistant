<!-- source: 04-training.html -->
<!-- index-title: 4: Tuesday -->

# DAT255 – DAT255: Deep learning engineering
# DAT255: Deep learning engineering

Lecture 4 – Training deep neural networks

sma@hvl.no

---

## The parameters of a neural network

---

## The activation function

What separates a neural network from standard linear regression is the non-linear **activation function**.

(And not that we have many layers – stacking linear functions is still a linear function)

\[
\small
\begin{aligned}
\color{DarkBlue}{f}(x) &= 2x + 3  & (\mathrm{linear}) \\
\color{Purple}{g}(x) &= 5x -1 & (\mathrm{linear}) \\
\color{DarkBlue}{f}(\color{Purple}{g}(x)) &= 2(5x-1) +3 &  \\
&= 10x + 1 & (\mathrm{also\;linear})
\end{aligned}
\]

---

## Activation function

A good activation function is

- Nonlinear
- Differentiable*
- Switches from **off** for negative inputs to **on** for positive inputs

Sigmoid

tanh

ReLU

GELU

Swish

*At least piecewise differentiable

---

## Last layer activation functions

In the final layer of the network, we need to choose an activation function that suits our task

- **Regression**: No activation – just sum the weighted connections. Output range \((-\infty, \infty)\) Also called *linear* activation. `keras.layers.Dense(units, activation=None)`
- **Classification**: Need something with output range \([0, 1]\) - Binary classification: Use **sigmoid** `keras.layers.Dense(1, activation='sigmoid)` - Multiclass: Use **softmax** `keras.layers.Dense(10, activation='softmax')` (remember one-hot encoding)

---

## Finding optimal parameters

For large networks we get a huge number of parameters

Need a clever way to optimise all at the same time.

The solution is **backpropagation**, which is relies on the network output being differentiable with respect to its parameters.

---

## Backpropagation

**Step 4**:

- Run one step of *gradient descent* to find better values for all parameters

---

## Gradient descent

With the gradient in place, we take steps downward (along the *negative* gradient), towards the optimal solution:

\[
\small
\boldsymbol{\color{teal}{\theta}}^{n+1} = \boldsymbol{\color{teal}{\theta}}^n - \eta \nabla \color{Purple}{L}
\]

Here \(\eta\) is the **learning rate**

-------------------------------------------------------------------------

---

## Learning rate

Learning rate is a hyperparameter

-------------------------------------------------------------------------

---

## Practical problems

Local
minimum
 `->` *bad predictions*

Plateau
 `->` *slow convergence*

Will try to solve these
problems next week

---

## Vanishing and exploding gradients

When backpropagation steps through the network layers, we can get unfortunate amplification effects:

1. **Vanishing** gradients:    Gradients go towards zero  no learning
2. **Exploding** gradients:    Gradients go towards infinity  no learning

Get improved training stability if we can make the variance of the *output* of a layer to be similar to the variance or the *input*:

---

## Some tricks

Common apporaches to avoid vanishing/exploding gradients:

- Choose a non-saturating activation function (like ReLU)

- Initialise each layer’s parameters according to number of input and output connections

- Add **normalisation layers** to the model (this is in Ch. 9 (next week) but metioning it now already)

---

## Normalisation layers

Most common: `keras.layers.BatchNormalisation`

From the documentation:
*(read the textbook for further details)*

> Batch normalization applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.
>
>
> Importantly, batch normalization works differently during training and during inference.

```
keras.layers.BatchNormalization(
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer="zeros",
    gamma_initializer="ones",
    moving_mean_initializer="zeros",
    moving_variance_initializer="ones",
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    synchronized=False,
    **kwargs
)
```

Comes with sensible defaults

---

## Regularisation

The usual L1 and L2 regularisation can be applied to neural network nodes

Technically three different options for where to add it (see docs):

```
layer = keras.layers.Dense(
    units=64,
    kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
    bias_regularizer=keras.regularizers.L2(1e-4),
    activity_regularizer=keras.regularizers.L1(1e-5)
)
```

---

## Dropout regularisation

The `rate` adjusts the percentage of nodes dropped

```
keras.layers.Dropout(rate, noise_shape=None, seed=None, **kwargs)
```

After convolutional layers it is recommended to rather use *spatial dropout*, which drops entire feature maps

```
keras.layers.SpatialDropout2D(
    rate, data_format=None, seed=None, name=None, dtype=None
)
```

---

## Putting together an improved network

```
from keras.layers import (
  Input, Rescaling, Conv2D, BatchNormalization,
  MaxPooling2D, Activation, Dropout, Dense
)

model = keras.Sequential([
  keras.Input(shape=(128, 128, 3)),
  Rescaling(1.0 / 255),
  Conv2D(128, kernel_size=3, kernel_initializer="he_uniform", padding="same"),
  BatchNormalization(),
  Activation("relu"),
  MaxPooling2D(3, padding="same"),
  # ...
  # (more layers)
  # ...
  Conv2D(128, kernel_size=3, kernel_initializer="he_uniform", padding="same"),
  BatchNormalization(),
  layers.Activation("relu"),
  MaxPooling2D(3, padding="same"),
  Flatten(),
  Dropout(0.3),
  Dense(num_classes, activation="softmax"),
])
```

---

## Best practices

With the choice of

- Architecture (layers and nodes)
- Parameter initialisers
- Activation functions
- Regularisation
- Optimisation algorithm
- Learning rate and scheduling

the search space for finding the optimal solution is huge

Guidelines from this and next week are meant to save time, but are not absolute.

*(Better guidelines will come along the next few years anyway)*
