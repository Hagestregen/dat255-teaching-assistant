<!-- source: 23_tokenisers.ipynb -->

# Tokeniser algorithms

In this notebook we get a little more serious about tokenisation.

While splitting text on whitespaces is a good start to get individial tokens, all modern language models split on subwords. So while "indivisible" is written as a single word, GPT-4 will see it as three separate tokens: "ind", "iv", and "isible". To test out how different LLMs implement their tokenisation step, have go at the tokeniser [playground](https://huggingface.co/spaces/Xenova/the-tokenizer-playground).

If we are to make our own language model, we can either make a custom tokeniser, or use a pretrained one. Let's investigate both options.

Modern tokenisers are actually running the text through a pipeline of several steps:

1. **Normalisation**: Clean the text by replacing diacritics, accents, and potentially convert to lower-case.
2. **Pre-tokenisation**: Do a first split of the text into smaller pieces -- typically whitespace-separated words.
3. **Subword tokenisation**: The difficult part -- find good ways to split words into subwords that can be combined in different ways.
4. **Post-processing**: Sometimes we want to insert special tokens, like "start-of-sentence \[SOS\]" or "end-of-sentence \[EOS\]". This is done in the post-processing step.

We can get pretrained tokenisers from several useful Python libraries, such as Keras Hub, TensorFlow Text, or Hugging Face. For this notebook we try the ones provided by Hugging Face [Tokenizers](https://huggingface.co/docs/tokenizers/en/index).


```python
import keras
```


Download some data. In this case, we use the _Wikitext-103_ dataset, which contains selected articles downloaded from WikiPedi.


```python
! wget -N https://wikitext.smerity.com/wikitext-103-raw-v1.zip
! unzip wikitext-103-raw-v1.zip
```


Check what it looks like:


```python
! head wikitext-103-raw/wiki.train.raw
```


## Normalisation

Depending on which language your data is written in, you might get a lot of "non-standard" characters that should be replaced or removed. The approach to do so is standised in a very serious fashion: https://unicode.org/reports/tr15

Luckily this is implemented for us in `tokenizers.normalizers.NFD()`. If we want to add multiple normalisation methods, we can do that in a sequential manner, like so:


```python
import tokenizers
from tokenizers.normalizers import NFD, StripAccents

normalizer = tokenizers.normalizers.Sequence([NFD(), StripAccents()])
```


Try it out:


```python
normalizer.normalize_str(
    'Here is sômè fünnÿ tẽxt with bôth Norwegiån (øæå), Greek (διακρίνω) and Arabic (هِجَائِيّ) characters.'
)
```


## Pre-tokenisation

Let's start by splitting text into words. The most common word dividers are of course spaces and newlines, but sometimes we also have contractions of two words that should be split.


```python
from tokenizers.pre_tokenizers import Whitespace
pre_tokenizer = Whitespace()
pre_tokenizer.pre_tokenize_str("Hey, what's up?")
```


Note that the `Whitespace` class will also split on punctuation, and returns the positions of each word in the original sentence.


Maybe we also wany to split numbers into separare digits?


```python
from tokenizers.pre_tokenizers import Digits
pre_tokenizer = tokenizers.pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])
pre_tokenizer.pre_tokenize_str("Call 911!")
```


## BPE subword tokenisation

Let's first try out the tokenisation algorithm used by OpenAI's GPT models: _Byte-Pair Encoding_ (BPE). For a tutorial (with video) on the details about bytepair encoding for tokenisation, have a look at the Hugging Face NLP [tutorial](https://huggingface.co/learn/nlp-course/en/chapter6/5).


```python
gpt_tokenizer = tokenizers.Tokenizer(BPE())
```


A special thing about this one, is that it builds spaces into the beginning of tokens, in case the token is the start of a word. This is part of the pre-tokenisation -- let's see how it works:


```python
gpt_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)

gpt_tokenizer.pre_tokenizer.pre_tokenize_str("Let's test byte-pair encoding pre-tokenization!")
```


We see spaces are encoded as the character "Ġ".


Now we can start training the tokeniser on our dataset. The `tokenizers` library comes with a `BpeTrainer` class, where we can set options such as the vocabulary size. Note that the training process can be memory hungry, so while a vocabulary size of 40-50k is common, we dial it down to avoid out-of-memory crashes. You can try to turn it up as far as it goes.

The GPT family of models technically don't use any normalisation, but we can add it anyway, just for good measure.


```python
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, Lowercase, StripAccents


gpt_tokenizer.normalizer = tokenizers.normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

gpt_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)


from tokenizers.trainers import BpeTrainer
trainer = BpeTrainer(vocab_size=10000, special_tokens=["<|endoftext|>"])
files = [f"wikitext-103-raw/wiki.{split}.raw" for split in ["train"]]#, "train", "valid"]]
gpt_tokenizer.train(files, trainer)
gpt_tokenizer.save("bpe-wiki.json")
```


Let's try out the trained tokeniser:


```python
output = gpt_tokenizer.encode("This is my awesome tokenised text. If it contains emojis, will it still work 🤨?")
print(output.ids)
# [1, 27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35, 2]
#gpt_tokenizer.decode([1, 27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35, 2])
# "Hello , y ' all ! How are you ?"
```


```python
gpt_tokenizer.decode(output.ids)
```


Try the real decoder:


```python
gpt_tokenizer.decoder = tokenizers.decoders.ByteLevel()
gpt_tokenizer.decode(output.ids)
```


To visualise how the text is split into subwords, we can use the `EncodingVisualizer`:


```python
from tokenizers.tools import EncodingVisualizer

viz = EncodingVisualizer(gpt_tokenizer)
viz('Here is some text to visualise. We have some eloquent words, some boring words, and also some typgin eroorrs.')
```


Next, we let you give it a go with the WordPiece algorithm, which was made popular with the BERT family of language models.
There is a tutorial and video about this method too: https://huggingface.co/learn/nlp-course/en/chapter6/6.

### <span style="color: red;">Exercise:<span>

Train a WordPiece tokeniser, and make the same visualisation as above, to identify similarities and differences.

