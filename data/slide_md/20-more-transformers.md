<!-- source: 20-more-transformers.html -->
<!-- index-title: 20: Tuesday -->

# DAT255 – DAT255: Deep learning engineering
# DAT255: Deep learning engineering

Lecture 20 – Multimodal transformers

sma@hvl.no

---

## Training LLMs

Typical training procedure:

1. **Self-supervised pre-training** A general model that can be subsequently fine-tuned on different tasks is called a *foundation* model Cost: $1 million to $100 million

1. **Supervised fine-tuning** Adapt to more specific uses, such as - Chat / Question answering - Chain-of-thought reasoning - Domain-specific used Cost: Depends

1. **Continuous fine-tuning** Fine-tune an already fine-tuned model Cost: $1 to $1000

---

## Training data

As always, key to a good model is data.

**Pre-training:**

Data scraped from …

- Wikipedia *(multi-language)*
- GitHub *(code)*
- ArXiv *(academic text)*
- StackOverflow *(Q&A)*
- Reddit etc. *(forums)*
- Project Gutenberg *(books)*

See e.g. Common Crawl

**Fine-tuning:**

Now we need ***annotated*** data

- Specific datasets for question answering
- Reinforcement learning from human feedback (RLHF)

Exact training data are business secrets (even for open-sourced models)

---

## Distributed training

Once the model can’t fit on a single GPU, things get more complicated

*(for instance the full Llama 3.1 needs 3.3TB VRAM for training)*

What can be split and parallelised?

- **Data:** Run different batches in parallel
- **Weights:** Distribute weight matrices over separate GPUs
- **Layers:** Distribute different layers over separate GPUs
- **Sequences:** Partition the input data sequences

Update entire model after each training step

TensorFlow implements a set of methods for distributed training, while Hugging Face offers advanced ones

---

## Supervised fine-tuning

A pre-trained model can only append text to an input.

Making a useful chatbot requires *instruction* training

```
{
  "instruction": "Translate 'Good night' into Spanish.",
  "solution": "Buenas noches"
}
{
  "instruction": "Name primary colors.",
  "solution": "Red, blue, yellow"
}
```

 For this step, high quality data is preferred to quantity.

On the Hugging Face model hub you will often find two variants of the same model:

`meta-llama/Llama-3.2-3B`
 `meta-llama/Llama-3.2-3B-Instruct`

> The Llama 3.2 collection of multilingual large language models (LLMs) is a collection of pretrained and instruction-tuned generative models in 1B and 3B sizes (…)

---

## LLM fine-tuning with limited resources

*Full* model fine-tuning can be problematic for two reasons:

- Hardware requirements (mainly VRAM)
- Risk of *catastrophic forgetting* (https://arxiv.org/abs/1312.6211)

Useful techniques:

- **Prompt tuning:** Add a small trainable model *before* the LLM, which outputs learned, task-specific tokens

- **Low-rank adaptation** (*LoRA*): Add small traininable layers *in parallel* with the existing attention layers

---

## Low-rank adaptation (*LoRA*)

Keep original weight matrix \(\small W\) frozen

Train new (small) matrices \(\small A\) and \(\small B\)

- Adds \(\small 2\cdot R\cdot D\) new parameters (compared to \(\small D^2\) in \(\small W\))

---

## Quantization

Reduce memory cost of running inference by reducing numerical precision in weights and activations

- Can store `float32` as `float16` without much modification
- Can store `float32` as `int8` for use on embedded systems (requires more modification)

Modern quantisation schemes for LLMs are more extreme:

Go down to anywhere between **6** to **2 bit**

*Example*:

`DeepSeek-R1-Q4_K_M.gguf`

`Q`{*bits per weight*}`_`{*type*}`_`{*variant*}

Scheme
Compression ratio
(relative to `f32`)
Performance

`Q8_0`
1:4
High quality

`Q4_K_M`
1:8
Medium quality

`Q3_K_M`
1:11
Low quality

---

## Knowledge distillation

---

## Transformers for other data types

---

## General transformers

Motivation behind transformers was to process sequential language data

Turns out they are great general-purpose models

- Make very few assumptions about the input data

Transformers are now among state-of-the-art for large-scale models on

- Text
- Images
- Video
- Audio
- Point clouds

In context of LLMs we call this
 different *modalities*

---

## Visual attention

Use the attention matrix as an explanation tool:

Mistakes :/

---

## Transformers for computer vision

Challenge for processing non-textual inputs:

How to define a token?

*Simplest approach:* Each pixel is a token.

Problem: Attention matrix is quadratic in number of tokens

for big images, the matrix becomes huge

Two common approches:

- Cut the image into patches
- First apply convolutional layers

---

## Vision transformer (ViT) for classification

Split the image in \(\small N\) patches: Instead of

height \(\small\times\) width \(\small\times\) colour channels

number of tokens, we get

\(\small N \times\) (patch size)\(^2\) \(\small\times\) colour channels

tokens

For each patch:

- Flatten it to an 1D vector
- Make embeddings
- Add positional embeddings

Treat it as any generic sequence and input to a transformer encoder

---

## Vision transformer (ViT) for classifiction

https://arxiv.org/pdf/2010.11929

---

## Positional embeddings

Still need to introduce the position of each patch

- *Option 1:* **Handcrafted encodings**, like the sinusoidal encoding of the original encoder - Gets a little complicated and generally does not perform very well

- *Option 2:* **Learned embeddings** - Instead of embedding word 1, 2, … \(\small N\), we trivially extend the procedure to embed patch (1,1), (2,1), … (\(\small N, M\)) `keras.layers.Embedding` can be used for any input dimensionality

---

## Combined architecture

https://arxiv.org/pdf/2005.12872

---

## Inductive biases

***Inductive bias:*** assumption built into the model architecture

- Linear models (\(\small y = ax+b\)): Assume data is linear
- CNNs: Assume translational equivariance, hierarchical structure
- RNNs: Assume meaningful ordering

---

**Transformers**: Assume *some* relation between input tokens, but that’s about it

*Good*: General-purpose architecture

*Bad*: Requires (a lot) more training data than problem-specific models

---

## Combining vision and text

---

## Audio transformers

Speech synthesis: Use an example (*acoustic prompt*) to determine generated style and tone

https://arxiv.org/abs/2407.08551

---

## Transformers for other uses

 https://deepmind.google/discover/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/

---

## Multimodal transformers

Transformer encoder-decoder structure allows for a single model to operate on selveral different types of data

If it can be encoded into an embedding space, it can be used as a context for a decoder.

---

SHOW:
- whisper
- alphastar
- visual question answering gemini
