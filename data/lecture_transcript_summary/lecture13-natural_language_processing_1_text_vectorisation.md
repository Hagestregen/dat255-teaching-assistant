# natural language processing 1: text vectorisation

The next three weeks focus on deep learning applied to image inputs, with tasks including image classification, object detection with bounding boxes, and person counting.

The project deadline is set for April 30, with two months allocated for completion. Data and computing resources are required, with options including Google Colab or a dedicated computing setup. Discussions are ongoing with Sigma 2 for additional resources.

The course covers four weeks of natural language processing (NLP), starting with simpler text processing tasks and progressing to text generation. A mini chat GPT model will be developed in the following weeks.

## Natural language processing (NLP)

The upcoming lectures will concentrate on deep learning techniques in natural language processing (NLP), particularly for image inputs. NLP involves processing human language, which is distinct from machine language. This field is crucial as it encompasses tasks such as image classification, object detection with bounding boxes, and person counting.

Historically, NLP has been around for decades, with significant advancements in the last few years due to deep learning. While older techniques are acknowledged, the emphasis will be on modern approaches.

The challenge of teaching computers language rules, despite our extensive knowledge, has led to the exploration of statistical inference methods. These methods enable machines to learn from text data, as demonstrated by models like ChatGPT, which generate coherent text without explicit programming.

The lectures will focus on acquiring skills in NLP through deep learning, bypass

## NLP tasks

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

In natural language processing (NLP), typical tasks involve text classification, sentiment analysis, summarization, and question answering. Classification involves categorizing text into predefined classes, such as identifying spam emails or determining the sentiment of a text. Sequence-to-vector problems are common in classification, where a sequence of data is mapped to a single output.

Sequence-to-sequence tasks, like text translation, involve converting text from one language to another, which is not a one-to-one word mapping due to language differences. Sentiment analysis determines the positive or negative tone of a text. Summarization condenses long texts into shorter versions that capture the main points. Question answering requires

## Our toolbox so far:

The next week's focus will be on advancing natural language processing techniques, particularly in the realm of image classification and reading comprehension. Embeddings are essential for these tasks, and the LLP model from 2017 serves as a foundational example. The Stanford AI Index indicates that while image classification has surpassed human performance since 2015, reading comprehension lagged behind, reaching only 80% of human capability in 2017. However, significant progress is anticipated in the upcoming week, with the aim to enhance models to match or exceed human performance in reading comprehension.

## Getting text into our model

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.


The process for text classification begins with standardization, converting text to lowercase and removing punctuation. Tokenization then turns words into tokens, with each word typically being a token. Tokens are assigned indices, with common words receiving smaller indices. One-hot encoding represents tokens as columns indicating their presence, but this method is inefficient for large dictionaries. Embeddings are used instead, representing words as vectors to maintain semantic relationships and improve efficiency.

## Text vectorisation

The Keras library includes a Texts Vectorization layer for converting text into numerical form. This layer transforms text into integers by mapping tokens to numbers, with options to limit the vocabulary size and standardize text by removing punctuation and converting to lowercase. The output mode can be either integer or one-hot encoding. The layer can be integrated into a model and adapts to the dataset to learn the vocabulary. In the context of the IMDB dataset, this layer helps classify movie reviews by encoding text into a numerical format that can be processed by machine learning models.

## Fancier text vectorisation

The transcript discusses the concept of tokenization in natural language processing (NLP), where words are broken down into tokens for processing. It highlights the challenge of handling linguistic variations like conjugations and the need for a fixed vocabulary to prevent memory overflow. Modern deep learning tools address this by employing subword tokenization, which splits words into smaller units, allowing for the representation of words with different modifiers while maintaining their semantic meaning.

## Tiktokenizer

The transcript focuses on the tokenization process in modern language models, highlighting the use of subwords for splitting text into indices. It demonstrates how different models may tokenize the same text differently, with some splitting by individual letters and others by meaningful word groups. The author notes the advantage of subword tokenization in handling unknown words, as it avoids assigning them an index number. The discussion suggests that while splitting by subwords is effective, grouping words together can also be a viable approach.

## Detour: Bag-of-words

The lecture will explore the application of deep learning to image inputs over the next three weeks, with a focus on computer vision tasks such as image classification, object detection with bounding boxes, and person counting.

In the context of natural language processing (NLP), the lecturer discusses simplifying the problem of text classification by treating tokens as indices rather than semantic entities. This approach disregards the meaning of individual words and their relationships, suggesting that a sequence of words may not be necessary for classification. The lecturer proposes using a simple model that classifies text based on the presence of certain keywords, such as "awful" for negative reviews and "favorites" for positive ones, without considering the order of words.

## Detour: N-grams

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

In the context of natural language processing, n-grams are used to simplify the problem of language modeling by considering pairs (bigrams) or triplets (trigrams) of words. This approach can efficiently handle certain tasks without the need for large language models. Bigrams and trigrams are sequences of two and three words, respectively, that capture local word order. While n-grams are less useful for capturing long-range dependencies in text, they are still relevant in other sequencing tasks such as DNA sequencing. In contrast, the bag-of-words model disregards word order entirely. More advanced techniques aim to preserve the entire sequence of words to better understand context and meaning.

## Sequence models

The upcoming weeks will focus on Transformers for image inputs, following the introduction of convolutional and recurrent neural networks. Practical exercises with Transformers will be conducted to assess their effectiveness.
