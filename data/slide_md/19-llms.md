<!-- source: 19-llms.html -->
<!-- index-title: 19: Monday -->

# DAT255 – DAT255: Deep learning engineering
# DAT255: Deep learning engineering

Lecture 19 – More about LLMs

sma@hvl.no

---

## Training decoder models

Decoder-type language models are next word predictors.

So we train them to do exactly this:

1. Mask the end of sentences
2. Use the next token as the prediction target
3. Reveal token and move to next one
4. Continue until end of text

Usually call the procedure *masked attention* or *causal attention*, when model is only allowed to look “backwards”.

Can train on large, unlabelled data in a self-supervised approach

---

## Generating text from a decoder model

Sampling strategies for next token:

- **Greedy search:** Always select token with highest score
- **Beam search:** Keep track of several possible branches of output sequences, and select the sentence with highest probability
- **Sampling:** Use scores as probabilities and sample randomly
- **Top-*K* sampling:** Sample among the *K* tokens with highest score
- **Adjusted softmax sampling:** Add a parameter *T* called temperature in the softmax function applied to the output Adjusts how tokens are sampled

---

## Softmax with temperature

Divide the logits *a* by temperature *T* in the softmax function:

\[
\small
y_i = \frac{\exp(a_i/T)}{\sum_j \exp(a_j/T)}
\]

Changes the distribution we sample from.

- **Low *T*:** Give highest score to the most likely token - More determinism

- **High *T*:** Give more equals scores to all tokens - More randomness / creativity

---

## Large language models (LLMs)

***LLM:*** language model with \(\small\gtrsim 10^9\) (1 billion) parameters

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

SHOW:
- whisper
- alphastar
- visual question answering gemini
