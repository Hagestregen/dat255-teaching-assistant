<!-- source: 13-nlp-basics.html -->
<!-- index-title: 13: Monday -->

# DAT255 – DAT255: Deep learning engineering
# DAT255: Deep learning engineering

Lecture 13
 Natural language processing 1: Text vectorisation

sma@hvl.no

---

## Natural language processing (NLP)

(as opposed to *machine language*)

Since most human knowledge is stored as text, **NLP** is an important field of study

> Knowing grammar, syntax and language structure (i.e. *linguistics*), can we write down the set of rules required to do a language task like translation?

Not really, no.

> Knowing basically nothing, but having a library of example text, can we statistically infer the rules required to do a language task?

Yes!

*Disclaimer:*
 There’s loads of interesting NLP stuff that we will skip, because it has been obsoleted by transformer models. Look at the supplemental reading on Canvas.

---

## NLP tasks

- Text classification: *What is the topic of this text?*
- Content filtering: *Is this email spam? Does this post contain swearing?*
- Sentiment analysis: *Is this review positive or negative?*
- Translation: *What is this text in French?*
- Summarisation: *Give a short summary of this article.*
- Question answering: *How do I fix the Wifi on my Thinkpad?*

We start off with classification, before moving to text output next week.

---

## Our toolbox so far:

We know how to do

- **Sequence processing:** Know how to construct RNNs, CNNs
- **Input encoding:** Looked at *embeddings* last time

With this, we can build an **NLP model at 2017 level**
 (will do so this week)

Next week we will learn about

- **Transformers** (not the electrical kind)

With this, we can build an **NLP model at 2022 level**

 ![[MMLU](https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu) benchmark](figures/lecture13/mmlu.png)

---

## Getting text into our model

**Text vectorisation**: Converting text to numeric data.

Typical approach consists of several steps

1. ***Standardisation:*** Remove diacritics, punctuation, convert to lowercase
2. ***Tokenisation:*** Split text into ***tokens*** which can be words, subwords, or groups of words
3. ***Indexing:*** Convert tokens to integer values
4. ***Encoding:*** Convert indices into embeddings or one-hot encoding

 [_Deep Learning with Python_, F. Chollet]{.absolute bottom="-2%" right="5%" style="font-size: 0.5em;"}

---

## Text vectorisation

Test sentence from the IMDb movie review dataset:

```
Text:     I   am  shocked.  Shocked  and  dismayed  that  the   428   of   you   IMDB  users
Encoded: 10  238     2355      2355    3         1    12    2     1    5    23    933   5911
Decoded:  i   am  shocked   shocked  and     [UNK]  that  the  [UNK]  of   you   imdb  users
```

Can remove HTML tags (or other uninteresting stuff) by using regexes:

```
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  without_html = tf.strings.regex_replace(lowercase, '<[^>]*>', ' ')
  without_punctuation = tf.strings.regex_replace(without_html, '[{}]'.format(string.punctuation), '')
  return without_punctuation

vectorize_layer = keras.layers.TextVectorization(
  standardize=custom_standardization
)
```

*Note:* `TextVectorization` is based on TensorFlow operations, so using it with other backend frameworks is complicated.

---

## Fancier text vectorisation

Word vectorisation is fast, but

- Could be more efficient (consider *small* - *small**er*** - *small**est***, *loud* - *loud**er*** - *loud**est***)
- Struggles with out-of-vocabulary words

Modern LLMs use **subword tokenisation** algorithms:

- Start out with individual characters
- Then merge them into subwords based on frequency of character combinations

---

*[Interactive slide: Tiktokenizer]*

---

## Detour: Bag-of-words

So far all words receive a separate index; we assume no relation between them.

Since text is not strictly ordered, how about training a `Dense` model on the *presence* of words, and just ignore the ordering?

Consider the IMDb review dataset again:

Review
Sentiment

Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to …
positive

If you like original gut wrenching laughter you will like this movie. If you are young …
positive

This movie made it into one of my top 10 most awful movies. Horrible. There wasn’t …
negative

Can classify decently well by learning correlations between single words and sentiment.

In this case we treat the text as a ***set*** and not a ***sequence***.

---

## Detour: N-grams

 ## Text as _sequences_ vs _sets_

N-grams apply to other data as well, for instance

Sequence
1-gram
2-gram
3-gram

Protein sequencing
Cys-Gly-Leu-Ser-Trp
Cys, Gly, Leu, Ser, Trp,
Cys-Gly, Gly-Leu, Leu-Ser, Ser-Trp
Cys-Gly-Leu, Gly-Leu-Ser, Leu-Ser-Trp

DNA sequencing
AGCTTCGA
A, G, C, T, T, C, G, A
AG, GC, CT, TT, TC, CG, GA
AGC, GCT, CTT, TTC, TCG, CGA,

For language research purpuses N-grams are still useful; have a look at
the Google N-gram Viewer

---

## Sequence models

Let’s put the ordering back in, and look at *sequence* models instead.

Now need a model type that assumes a relation between neighbouring words

We know of two:

- Convolutional neural networks (CNNs)
- Recurrent neural networks (RNNs)

For NLP uses, both are superseeded by *transformers*, which is the topic for next week.

But let’s go through a trick tomorrow that improves performance for all model types: word *embeddings*.
