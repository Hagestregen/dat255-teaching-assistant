<!-- source: 05-optimisers.html -->
<!-- index-title: 5: Monday -->

# DAT255 – DAT255: Deep learning engineering
# DAT255: Deep learning engineering

Lecture 5 – Optimisers and loss curves

sma@hvl.no

---

## This week

**Today:**

- A little more on gradient descent - Improved optimisation methods
- Loss curves and learning rates

**Tomorrow:**

- Data augmentation
- Non-sequential layer connections
- Training *very* deep convolutional networks
- Modern ConvNet architectures

---

## Loss functions

*Reminder:* For all statistical modelling we need a measure of prediction quality

The **loss function** \(\small{}L\) must satisfy two conditions:

- Differentiable*
- Bounded below (\(\small{}L \geq 0\))

In this course we stick mostly to

- **Mean squared error (MSE)** loss, for regression tasks
- **Cross-entropy** loss (or log loss), for classification tasks

\[
\scriptsize
L_{\mathrm{log}}(\boldsymbol{\hat{y}, y}) = -\frac{1}{N} \sum_i^N \sum_k^K y_{i,k} \log\hat{y}_{i,k}
\] where \(K\) is the number of classes

*Again, at least piecewise differentiable

---

## Gradient descent

With the gradient in place, we take steps downward (along the *negative* gradient), towards the optimal solution:

\[
\small
\boldsymbol{\color{teal}{\theta}}^{n+1} = \boldsymbol{\color{teal}{\theta}}^n - \eta \nabla \color{Purple}{L}
\]

Here \(\eta\) is the **learning rate**

---

## Local minima

Local
minimum
 `->` *bad predictions*

---

*[Interactive slide: Keras optimisers]*

---

## Comparing gradient descent algorithms

---

## Momentum optimisation

Improve on regular gradient descent by keeping track of *past gradients* in a new term \(\small\boldsymbol{m}\):

\[
\small
\begin{align}
    \boldsymbol{m}_n &= \beta \boldsymbol{m}_{n-1} - \eta \nabla \color{Purple}{L}(\color{teal}{\theta}_n) \\
    \boldsymbol{\color{teal}{\theta}}_{n+1} &= \boldsymbol{\color{teal}{\theta}}_n + \boldsymbol{m}_n
\end{align}
\]

(behaves like a heavy ball rolling down a hill, where \(\small\boldsymbol{m}\) is the velocity)

Introduces a new hyperparameter \(\small\beta \in [0,1]\), which is called the momentum

```
keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
```

---

## Nesterov accelerated gradient

Keep the concept of momentum, but compute the gradient *slightly ahead*:

\[
\small
\begin{align}
    \boldsymbol{m}_n &=
        \beta \boldsymbol{m}_{n-1}
        - \eta \nabla \color{Purple}{L}(\color{teal}{\theta}_n + \beta \boldsymbol{m}_{n-1}) \\
    \boldsymbol{\color{teal}{\theta}}_{n+1} &= \boldsymbol{\color{teal}{\theta}}_n + \boldsymbol{m}_n
\end{align}
\]

```
keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
```

---

## AdaGrad (*Ada*ptive *Grad*ient)

AdaGrad introduces an **adaptive learning rate**, which is adjusted *independently* for the different parameters

- For parameters with steep gradient, learning rate is reduced quickly
- For parameters with shallow gradient, learning rate is reduced slowly

AdaGrad in practice:

- Less sensitive to choice of learning rate (since it’s adaptive) (*good*)
- Fast convergence for simple problems (*good*)
- Slow (or no) convergence for difficult problems (*bad*)

Add intelligent scaling to get **RMSProp**

---

## Adam (and friends)

Momentum

+ RMSProp

+ some technical details

= **adaptive moment estimation (Adam**)

```
keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
```

 [_(For max confusion, the hyperparameters now changed names to $\beta_1,\,\beta_2$)_]{style="font-size: 0.6em;"}

More tricks:

- **AdaMax**: Scale the past gradients differently
- **Nadam**: Add Nesterov momentum
- **AdamW** Add L2 regularisation (*aka* weight decay)

:::{.fragment fragment-index=2}
:::{.absolute bottom="0%" right="10%" width="150px"}
![](figures/lecture5/allthethings.png)
:::
:::

---

## Comparison and guidelines

Method
Convergence

Speed
Quality

SGD
⚡️
⭐️⭐️⭐️

SGD w/ momentum
⚡️⚡️
⭐️⭐️⭐️

SGD w/ Nesterov
⚡️⚡️
⭐️⭐️⭐️

AdaGrad
⚡️⚡️⚡️
⭐️

RMSProp
⚡️⚡️⚡️
⭐️⭐️(⭐️)

Adam
⚡️⚡️⚡️
⭐️⭐️(⭐️)

AdaMax
⚡️⚡️⚡️
⭐️⭐️(⭐️)

Nadam
⚡️⚡️⚡️
⭐️⭐️(⭐️)

AdamW
⚡️⚡️⚡️
⭐️⭐️(⭐️)

A. Geron: *Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow*

---

## Learning rate and loss curves

Common to all optimisation methods is that we must choose a learning rate \(\eta\).

This affects the training progress:

---

## Learning rate scheduling

Some options for the most efficient learning: (*see notebooks*)

- Reduce \(\eta\) when learning stops: ``` keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001) ```
- Gradually reduce \(\eta\) for each step: ``` keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps, decay_rate) keras.optimizers.schedules.PolynomialDecay(initial_learning_rate, ...) ```
- Change \(\eta\) by some other rule: ``` class MyLRSchedule(keras.optimizers.schedules.LearningRateSchedule): def __init__(self, initial_learning_rate): self.initial_learning_rate = initial_learning_rate def __call__(self, step): return self.initial_learning_rate / (step + 1) optimizer = keras.optimizers.SGD(learning_rate=MyLRSchedule(0.1)) ```

## Referansegruppe

:::{.absolute top="10%" left="-7%" width="120%"}
![](figures/lecture2/referansegruppe.png)
:::

:::{.absolute top="70%" left="80%"}
![](figures/lecture2/uncle-sam.jpg)
:::

---

## Learning rate schedulers
