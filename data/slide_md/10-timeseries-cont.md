<!-- source: 10-timeseries-cont.html -->
<!-- index-title: 10: Tuesday -->

# DAT255 – DAT255: Deep learning engineering
# DAT255: Deep learning engineering

Lecture 10 – Sequences and time series

sma@hvl.no

---

## Project

On Canvas:

- (Optional) Look at the project catalog
- Add your group
- Submit project proposal – deadline Feb. 27

---

*Next week:* **Project kick-off**

- We go through requirements and expectations
- Start planning  - Get help/hints/feedback   - Get project approval   - Start working!

*Week 9:* **Project work**

- Decide on topic, data, and models
- Get help/hints/feedback
- Get project approval
- Start working! (no ordinary lectures this week)

---

## Project submission date

**Suggestions:**

- April 10
- April 17
- April 24
- May 1

Link also here and on Canvas

---

## Sequences

Sequences are **ordered** series. For instance

- **Natural language** *I only drink coffee* (\(\neq\) *only I drink coffee*)

- **Time series** **Time** 11:00 12:00 13:00 14:00 15:00 **Temp** 7°C 8°C 10°C 12 °C 12 °C

- **Audio**

- **Video** Video

- **DNA**

---

## Recurrent neural networks (RNNs)

Recall that a regular `Dense` layer computes its output by

```
outputs = activation(tf.dot(W, inputs) + b)
```

(where `W` is the weight matrix, `inputs` is the vector of features and `b` is the bias vector)

The recurrent node has two sets of weights:

- The usual ones, call them `W_x`
- Those to be applied to the previous output, call them `U`

The outputs then become

```
state_t = tf.zeros(shape=(num_output_features))
outputs = []
for input_t in input_sequence:  # loop over inputs at time t
  output_t = activation(tf.dot(W_y, inputs) + tf.dot(U, state_t) + b)
  outputs.append(output_t)
  state_t = output_t
output_sequence = tf.stack(outputs, axis=0)
```

Implemented in Keras as `keras.layers.SimpleRNN`

## Input and output sequences

![](figures/lecture10/input-output-sequences.png){fig-align="center" width="800px"}

:::{.absolute bottom="0%" left="0%" style="font-size: 0.5em;"}
A. Geron: _Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow_
:::

---

## *Intermezzo:* Autoregressive models

More advanced forecast:

The value tomorrow is given by a weighted sum of the \(p\) previous time steps, plus a noise term

\[
\small
y_i = \sum^p \varphi_i y_{t-i} + \epsilon_t
\]

(\(\varphi_i\) are the parameters of the model)

Can add *moving average* to get an **ARMA** model, look at differences to get **ARIMA**, add seasonality to get **SARIMA**, …

Lots of work on (traditional) statistical time series modelling - usually worth trying out before going to deep learning.

---

## Improved *memory cells*

In practice, RNNs suffer from vanishing/exploding gradients during training

Difficult to make them learn long-term dependencies.

Can introduce hidden states which are ***not*** the same as the output.

Two most used approaces: **LSTM**s and **GRU**s.

---

## The *long short-term memory* (LSTM) cell

Add long-term memory by having two states in each cell:

A short-term state \(\small\boldsymbol{h}_t\) and a long-term state \(\small\boldsymbol{c}_t\)

Gates determine data flow – add small networks inside the cell to act as *gate operators*

`keras.layers.LSTM`

---

## The *gated recurrent unit* (GRU)

Simplified and somewhat more effective variant:

`keras.layers.GRU`

A. Geron: *Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow*

---

## Stacking recurrent layers

As usual, we can increase the capacity by stacking layers.

Note when building a ***deep*** RNN: Intermediate layers should return the entire sequence

```
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.GRU(32, return_sequences=True)(inputs)
x = layers.GRU(32, return_sequences=True)(x)
x = layers.GRU(32)(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
```

---

## Training RNNs

Some tricks to efficiently training recurrent networks:

- Use saturating activation functions (tanh, sigmoid) `layers.LSTM(units, activation="tanh", recurrent_activation="sigmoid")`
- Use layer normalisation (`keras.layers.LayerNormalization`) instead of batch normalisation
- Add recurrent dropout (potentially in addition to regular dropout) `x = layers.LSTM(32, recurrent_dropout=0.25)(inputs)`

- Test if training runs faster on CPU than on GPU - NVIDIA backend only available if using default arguments for `LSTM`/`GRU` layers - `for` loops in recurrent nodes reduces parallelisability - Can optionally *unroll* `for` loops (memory intensive): `x = layers.LSTM(32, unroll=True)(inputs)`

---

## Bonus trick #1: CNN processing

Even with the previous tricks up out sleeve, getting RNNs to learn patterns over >100 time steps is difficult.

Can extract small-scale patterns with convolutional layers first, then apply recurrent layers

***Or*** maybe skip the recurrence altogether? **WaveNet** architecture:

```
keras.layers.Conv1D(..., padding="casual")  # Look only backwards
```

---

## Bonus trick #2: *bidirectional* RNNs

For time series we expect the most recent data points to be most important

Chronological ordering makes sense

Sometimes this is not the case - for instance for text

> I arrived by bike.

> Ich bin mit Fahrrad angekommen.

Can process sequences both forwards and in reverse by using a *bidirectional* recurrent layer:

```
inputs = keras.Input(shape=(...))
x = layers.Bidirectional(layers.LSTM(16))(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
```
