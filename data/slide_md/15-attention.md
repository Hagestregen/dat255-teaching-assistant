<!-- source: 15-attention.html -->
<!-- index-title: 15: Monday -->

# DAT255 – DAT255: Deep learning engineering
# DAT255: Deep learning engineering

Lecture 15 – Attention and Transformers

sma@hvl.no

---

epoch.ai

---

## Sequence-to-sequence learning

Using RNNs:

Limitations:

- All of the source text must be stored in the hidden state `->` in practice we get limited to short texts
- RNNs struggle to maintain memory over long sequences `->` again we get limited to short texts

---

## Vector embeddings

Tokenise and embed the words:

“you are **right**”

---

## Transformers: Basic idea

Tokenise and embed the words:

“you are **right**”

“turn **right** here”

transform!

transform!

Embedding space

New representation

---

## Transformers: The basic idea

***Aim:***

- Take a sequence of vectors of dimensionality \(\small N \times D\)
- Compute some relation between the vectors
- Use these relations to transform the vectors into new ones (also \(\small N \times D\))
- New representation is better suited to solve the task

Critical operation is to incorporate the relation between vectors

`->` the ***neural attention*** mechanism

---

## Self-attention

Example sentence: *“The train left the station on time”*

Repeat
for all tokens

---

## Self-attention: The steps involved

**4: Create the new, context-aware vector**

Sum up attention-weighted token vectors

---

## Self-attention

In (slow) code, we can write it as

```
def self_attention(input_sequence):
  output = tf.zeros(shape=input_sequence.shape)

  for i, vector in enumerate(input_sequence):
    scores = tf.zeros(shape=(len(input_sequence), ))

    for j, other_vector in enumerate(input_sequence):
      scores[j] = tf.tensordot(vector, other_vector, axis=1)

    scores /= np.sqrt(input_sequence.shape[1])
    scores = tf.nn.softmax(scores)

    new_representation = tf.zeros(shape=vector.shape)
    for j, other_vector in enumerate(input_sequence):
      new_representation += other_vector * scores[j]

    output[i] = new_representation

  return output
```

Return the new vector representation

---

## The *query-key-value* model

We can generalise the attention mechanism by using concepts from information retrieval.

What we computed was basically

```
output = sum(inputs * pairwise_scores(inputs, inputs))
```

but we could be doing this with three different sequences:

```
output = sum(values * pairwise_scores(query, keys))
```

> For each element in a query, compute how much it is related to every key, and use these scores to weight a sum of values

---

## Multi-head attention

Now we want to increase the complexity by computing attention several times in parallel.

*But:* Our transformation so far has no learnable parameters

it’s just a stateless operation, we get the same result each time.

Key idea from the *“Attention is all you need”* paper:

- Pass each input (query, key and value) though a separate `Dense` layer - Each with their own parameters

This we can do in parallel, and get different features for each attention “head”

Call it **multi-head attention**.

##

Stop

## Multi-head attention

![](figures/lecture15/multi-head-attention.png){fig-align="center" width="950px"}

:::{.fragment}
:::{.absolute top="65%" left="7.5%" width="400px" height="80px" style="border: 3px solid #FF0000;"}
:::

:::{.absolute top="65%" left="53%" width="400px" height="80px" style="border: 3px solid #FF0000;"}
:::
:::

[_Deep Learning with Python_, F. Chollet]{.absolute top="0%" right="0%" style="font-size: 0.5em;"}

## Multi-head attention

::::{.columns}
:::{.column width="50%"}

The parameters of the `Dense` layers that form the Q, K, V matrices, is where the attention head actually [_learns_]{.color-teal-dark} something.

:::{.fragment}
If $\small D$ is the dimensionality of the embedding space, and $\small N$ is the number of tokens in our data matrix $\small X$,

this introduces 3 weight matrices $\small W$, of dimension $\small D \times D$:

$$
\small
\begin{align}
Q &= X W^{(q)} \\
K &= X W^{(k)} \\
V &= X W^{(v)} \\
\end{align}
$$
:::

:::
:::{.column width="50%"}
![](figures/lecture15/multi-head-attention.png){style="padding-top: 100px;" fig-align="right" width="450px"}
:::
::::

## Multi-head attention

:::{style="padding-top: 40px;"}
:::

::::{.columns}
:::{.column width="55%"}
In the end, we can write the scaled dot-product self-attention for a single head as

$$
\small
\mathrm{attention}(Q, K, V) = \mathrm{softmax}\left[\frac{QK^T}{\sqrt{D}} \right] V
$$

And to get the final output from _all_ the attention heads, we simply concatenate the individial outputs.
:::
::::

:::{.absolute top="10%" right="0%" width="300px"}
![](figures/lecture15/scaled-dot-prod.png)
:::

[_Deep Learning: Foundations and Concepts_, C. Bishop]{.absolute bottom="0%" right="0%" style="font-size: 0.5em;"}

## Multi-head attention

:::{style="padding-top: 40px;"}
:::

::::{.columns}
:::{.column width="55%"}
In the end, we can write the scaled dot-product self-attention for a single head as

$$
\small
\mathrm{attention}(Q, K, V) = \mathrm{softmax}\left[\frac{QK^T}{\sqrt{D}} \right] V
$$

And to get the final output from _all_ the attention heads, we simply concatenate the individial outputs.
:::
::::

:::{.absolute top="20%" right="0%"}
<iframe width="370px" height="600px" src="figures/lecture15/bertviz_head_view.html" style="frame-scale: 0.75;"></iframe>
:::

## Bonus feature

::::{.columns}
:::{.column width="50%"}
Before we started involving `Dense` layers and the Q, K, V matrices, we had this:

![](figures/lecture15/attention-zoom.png){width="400px"}
:::

:::{.column width="50%"}

:::{style="padding-top: 150px;"}
:::

:::{.fragment}
:::{.colorbox style="background-color: #E0F2F1;"}
Notice it's symmetric.

The output from

$$\small\mathrm{attention}(Q, K, V)$$

on the other hand, need **not** be symmetric.

:::{.fragment}
Can then encode asymmetric relations:

A **hammer** is a **tool**, but not all **tools** are **hammers**.
:::
:::
:::
:::
::::

## The Transformer architecture

:::{style="padding-top: 30px;"}
:::

::::{.columns}
:::{.column width="60%"}
With the multi-head attention in place, we add a few extra layers to form a block:

:::{.incremental}
- A [residual connection]{.color-deep-orange-dark} (_Add & Norm_) going around the attention layer
- A stack of `Dense` (_Feed Forward_) layers
- A [residual connection]{.color-deep-orange-dark} going around the feed forward layers
:::
:::
::::

:::{.fragment}
This _almost_ completes our transformer encoder.
:::

:::{.absolute top="10%" right="0%" width="400px"}
![](figures/lecture15/encoder-zoom.png)
:::

:::{.absolute top="70%" right="0%" width="200px"}
![](figures/lecture1/attention.png)
:::

## Positional encodings

So far, the position of each token in a sentence<br>doesn't affect the computation of the attention scores.

`->` If we permute the word order, we still get the<br>same result.

:::{style="padding-top: 10px; text-align: center;"}
[_The food was bad, not good at all._]{.color-red-dark}

$$
\small
\neq
$$

[_The food was good, not bad at all._]{.color-green-dark}
:::

:::{style="padding-top: 10px;"}
:::

:::{.fragment}
Transformer models solve this by **encoding token position into the data itself:**

[token embedding with position]{.color-indigo} = [token embedding]{.color-pink-dark} + [position encoding]{.color-deep-purple}
:::

:::{.absolute top="0%" right="0%" width="250px"}
![](figures/lecture15/attention-zoom.png)
:::

## Positional encodings

- Option 1: Count token positions, then embed them

|     |     |      |     |      |
| --- |:---:|:---:|:---:|:---:|
| _token_ | The | food | was | good |
| _token embedding_ | $\mathbf{x}_1$ | $\mathbf{x}_2$ | $\mathbf{x}_3$ | $\mathbf{x}_4$ |
| _token position_ | 1 | 2 | 3 | 4 |
| _position embedding_ | $\mathbf{r}_1$ | $\mathbf{r}_2$ | $\mathbf{r}_3$ | $\mathbf{r}_4$ |
| _final embedding_ | $\mathbf{x}_1 + \mathbf{r}_1$ | $\mathbf{x}_2 + \mathbf{r}_2$ | $\mathbf{x}_3 + \mathbf{r}_3$ | $\mathbf{x}_4 + \mathbf{r}_4$ |
: {tbl-colwidths="[40,15,15,15,15]"}

:::{style="padding-top: 20px;"}
:::

:::{.fragment}
- Option 2: _Sinusoidal_ encoding (see textbook p 613)

:::{.absolute left="15%" bottom="-5%" width="200px" }
![](figures/lecture15/Figure_10_a.png){style="transform: rotate(90deg);"}
:::

:::{.absolute right="15%" bottom="-5%" width="200px" }
![](figures/lecture15/Figure_10_b.png){style="transform: rotate(90deg);"}
:::

:::
