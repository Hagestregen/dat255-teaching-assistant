<!-- source: 16-transformers.html -->
<!-- index-title: 16: Tuesday -->

# DAT255 – DAT255: Deep learning engineering
# DAT255: Deep learning engineering

Lecture 16 – Transformers for natural language processing

sma@hvl.no

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

---

## Multi-head attention

In the end, we can write the scaled dot-product self-attention for a single head as

\[
\small
\mathrm{attention}(Q, K, V) = \mathrm{softmax}\left[\frac{QK^T}{\sqrt{D}} \right] V
\]

And to get the final output from *all* the attention heads, we simply concatenate the individial outputs.

---

## The Transformer architecture

With the multi-head attention in place, we add a few extra layers to form a block:

- A residual connection (*Add & Norm*) going around the attention layer
- A stack of `Dense` (*Feed Forward*) layers
- A residual connection going around the feed forward layers

This *almost* completes our transformer encoder.

---

## Positional encodings

- Option 1: Count token positions, then embed them

*token*
The
food
was
good

*token embedding*
\(\mathbf{x}_1\)
\(\mathbf{x}_2\)
\(\mathbf{x}_3\)
\(\mathbf{x}_4\)

*token position*
1
2
3
4

*position embedding*
\(\mathbf{r}_1\)
\(\mathbf{r}_2\)
\(\mathbf{r}_3\)
\(\mathbf{r}_4\)

*final embedding*
\(\mathbf{x}_1 + \mathbf{r}_1\)
\(\mathbf{x}_2 + \mathbf{r}_2\)
\(\mathbf{x}_3 + \mathbf{r}_3\)
\(\mathbf{x}_4 + \mathbf{r}_4\)

- Option 2: *Sinusoidal* encoding

---

## Computational complexity

- For each attention head, we introduce \(\small 3 \cdot (D^2 + D)\) new parameters (The \(W^{(q)}), W^{(k)}, W^{(v)}\) matrices)
- Then we add dense layers on top, adding another \(\small D^2 + D\) parameters per layer
- Layer normalisation adds yet another \(\small 2\cdot D\) parameters

Quickly adds up to a lot.

**Still:** Computational cost of a forward pass of a transformer model is

- \(\small \mathcal{O}(N^2 D)\) for the attention layers
- \(\small \mathcal{O}(N D^2)\) for the dense layers

Compare to \(\small \mathcal{O}(N^2 D^2)\) for a fully-connected dense network

Transformers are a lot more efficient for large models

---

## Transformer model structures

Typically divide by which parts of the original transformer architecture is being used:

- **Encoder** models: Useful for sentiment analysis and similar tasks
- **Decoder** models: Most modern language models are this type
- **Encoder-decoder** models: Sequence-to-sequence models, useful for translation and for multimodal tasks

---

## Encoder transformers

***Input:*** Class token and randomly masked sequence

***Output:*** Class prediction and completed sequence

Example: *Bidirectional Encoder Representations from Transformers* (BERT), 2019

Can be fine-tuned for various tasks, but:

- Not suited for text generation
- Training is inefficient

(`positive`) The `<masked>` was great (…)

(`93%` positive) The `film` was great (…)

https://arxiv.org/abs/1810.04805
 https://huggingface.co/docs/transformers/en/model_doc/bert

---

## Decoder transformers

Decoder-only transformers can be used as *generative* models

(like GPT: *Generative Pre-trained Transformer*, 2018)

***Input:*** A sequence

***Output:*** Approximate probabilites for the next token in the sequence

Examples:

- GPT (2018), GPT-2 (2019), GPT-3 (2020), …
- LlaMa versions (2023-2024)
- DeepSeek versions (2023-2025)
- And most other LLMs – although they differ in implementation and training

https://openai.com/index/language-unsupervised/
 https://huggingface.co/docs/transformers/en/model_doc/openai-gpt

---

## Encoder-decoder models

Original transformer model was a *sequence-to-sequence* model

Combining the encoder and decoder parts relies on cross-attention

The cross-attention mechanism is often used in *multi-modal* models such as speech-to-text or image-to-text

---

Yang, Jingfeng, et al. “Harnessing the power of LLMs in practice: A survey on ChatGPT and beyond.” *ACM Transactions on Knowledge Discovery from Data* 18.6 (2024): 1-32.

---

## Training data

As always, key to a good model is data.

Typically use data scraped from …

- Wikipedia *(multi-language)*
- GitHub *(code)*
- ArXiv *(academic text)*
- StackOverflow *(Q&A)*
- Reddit etc. *(forums)*
- Project Gutenberg *(books)*

See e.g. Common Crawl

---

https://huggingface.co/models

---

*[Interactive slide: Keras Hub]*

---

*[Interactive slide: Training data]*
