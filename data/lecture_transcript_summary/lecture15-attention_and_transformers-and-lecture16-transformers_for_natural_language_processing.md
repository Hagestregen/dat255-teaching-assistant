# attention and transformers and lecture 16: transformers for natural language processing

The upcoming weeks will delve into transformers, a pivotal neural network architecture underpinning models like ChatGPT. The initial focus will be on the foundational concept of attention. Following a brief hiatus, Tarja will resume lectures on Tuesday, advancing beyond the textbook to explore transformers' applications in various contexts. The curriculum will extend to include generative AI, with a particular emphasis on image generation, culminating in project work and exam preparations. The project submission deadline is set for April 30.

## Sequence-to-sequence learning

The next three weeks focus on deep learning applied to image inputs, with tasks including image classification, object detection with bounding boxes, and person counting.

Sequence-to-sequence learning is essential for translation tasks, where the entire sentence context is considered for generating outputs. Word-by-word translation fails due to varying word orders and the need for different numbers of words across languages. Encoding and decoding processes, using recurrent neural networks (RNNs) like LSTMs, are abstract but broadly applicable in machine learning. Embeddings are crucial for language tasks, with a simple approach involving embedding input and using LSTM for sequence generation. RNNs face limitations, such as scaling issues and sequential processing, which hinder parallelization on GPUs. For translation, RNNs can handle one sentence at a time, but context retention is

## Attention is all you need

The upcoming chapter will delve into the nuances of handling output probabilities in text generation, a critical aspect when developing systems like ChatGPT.

## Vector embeddings

"Last week's discussion on embeddings concluded with the concept of tokenizing words and representing them as vectors in a space. These vectors, despite the limitations of visualizing in three dimensions, effectively map tokens to points in the vector space."

## Transformers: Basic idea

The next three weeks focus on deep learning applied to image inputs, with tasks including image classification, object detection with bounding boxes, and person counting.

In deep learning, embeddings group similar words in proximity within a high-dimensional space. However, post-training, embeddings yield identical positions for different tokens, limiting contextual understanding. Transformers address this by adjusting token positions based on surrounding words, enhancing semantic differentiation.

Technically, embeddings are vectors in a space defined by sequence length \(n\) and embedding size \(d\), with \(n\) representing token count and \(d\) the embedding dimension. Transformers compute relations between tokens using distance or cosine similarity, transforming vectors to maintain meaningful directionality.

The attention mechanism, akin to human attention, enables models to focus on relevant features, facilitating tasks like classification. This mechanism, along with transformers, forms the basis for advanced deep learning models in image processing.

## Self-attention

The next three weeks focus on deep learning applied to image inputs, with tasks including image classification, object detection with bounding boxes, and person counting.

In discussing the word "station," it's noted that it has multiple meanings, such as petrol, meteorological, train, radio, or TV stations. Context is crucial for determining the specific type of station being referred to. Longer texts require examining various sections to understand the context. The challenge of context construction is significant, even for children or adults. The goal is to address this challenge.

The book's chapter 15, which was previously mentioned, is not the focus; instead, the emphasis is on self-attention mechanisms. Self-attention involves analyz

## Self-attention: The steps involved

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

In deep learning, each word in a sentence is represented as a token, which is then converted into a vector through an embedding process. The embedding layer transforms these tokens into vectors, capturing semantic meaning.

The attention mechanism computes scores between tokens to determine their relevance to each other within the context of the sentence. This process involves calculating the dot product of vectors, which reflects the similarity between tokens. The attention scores highlight the relationships between words, such as the connection between "station" and "train" in a given sentence.

After computing attention scores, the vectors are weighted and summed to produce new context-aware vectors for each token. These vectors incorporate information from the entire sentence, allowing the model to understand the context in which each word appears.

The self-attention mechanism simplifies the process of creating context-aware vectors, which are crucial for tasks like image classification and object detection in computer vision. The lecture discusses the application of deep learning to image inputs, focusing on computer vision tasks such as image classification, object detection with bounding boxes, and person counting. It explains the concept of orthogonality in vectors, where orthogonal vectors yield a zero dot product, indicating no relation. The lecture then introduces attention scores, which are computed to measure the relationship between vectors, with scores ranging from zero to one. These scores are scaled by the input length and normalized using a softmax function to ensure they fall within a fixed interval. The attention mechanism allows for the creation of new representations by weighting vectors with these scores and summing them up. The lecture highlights the importance of the position of words in a sequence, as the attention mechanism captures the relation between words regardless of their position. It also touches on the challenges of managing large attention matrices in memory, which contribute to rising RAM prices.

## The query-key-value model

The upcoming three weeks will delve into deep learning with a focus on image inputs. The curriculum will cover computer vision tasks such as image classification, object detection with bounding boxes, and person counting.

In the context of transformer models, pairwise scores between words are computed, and all possible pairs are considered. This process involves using the sequence of tokens to affect the vectors within the sequence, resulting in an output vector for each token. The input vector appears three times within the sequence, each time being modified by the corresponding score. This mechanism can be likened to a lookup operation, where values are scaled based on the match between the query and the keys, akin to probing a database.

The transformer model's attention mechanism can be visualized as a series of queries, keys, and values, where the values are scaled by the relevance of the query to the keys. This concept is rooted in information theory and is applicable across various fields. The full implementation of the transformer model will be demonstrated using a figure from the referenced material.

## Multi-head attention

The next three weeks focus on deep learning applied to image inputs, specifically in the realm of computer vision tasks such as image classification, object detection with bounding boxes, and person counting.

The attention mechanism, a pivotal concept in deep learning, allows models to focus on different parts of the input data, enhancing their ability to understand context and relationships within the data. Initially, attention computations were purely computational, relying on pre-learned embeddings without introducing new parameters. However, this approach was found to be stateless and lacking in learning capability.

To address this limitation, the introduction of multi-head attention was proposed. This technique involves adding dense layers to each instance, enabling the model to learn from the training data and adapt its focus based on the input. By performing the attention mechanism in parallel across multiple heads, the model can capture a more diverse set of patterns and relationships, leading to improved performance.

The multi-head attention mechanism involves several steps:
1. Computing the attention mechanism for each head separately.
2. Scaling the values based on the number of dimensions.
3. Applying a softmax function to normalize the values.
4. Concatenating the results from all heads.

This process results in a more complex but effective attention mechanism, allowing the model to generate a new output vector for each element in the sequence. The model learns to match queries against available keys and scale values accordingly, enhancing its ability to understand and process the input data.

The use of multi-head attention in deep learning models, particularly in the context of image inputs, represents a significant advancement in the field. By enabling models to learn and adapt their focus based on the input data, this technique improves the model's ability to perform complex tasks such as image classification, object detection, and person counting. The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

The GPT-2 model, with its 11 layers of attention and 12 attention heads, demonstrates the ability to associate words and concepts. For instance, the model associates "deep" with "learning," indicating an understanding of deep learning as a significant concept. It also recognizes punctuation patterns, such as the relationship between question marks and exclamation points.

Attention mechanisms in models like GPT-2 allow for the recognition of various patterns, which can be useful for tasks like punctuation pattern recognition. However, the model's performance varies across different layers and attention heads, with some layers showing less useful patterns.

Despite the complexity of large language models, their performance can be analyzed through attention scores and patterns. These models are not magic; their outputs are based on learned patterns and data. Attention mechanisms are a key component in understanding and interpreting these patterns.

## The Transformer architecture

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

The transformer model architecture incorporates attention mechanisms and residual connections, which are crucial for maintaining information flow and enabling the model to learn effectively. The attention mechanism processes contextual information, while dense layers perform pattern recognition and refinement. This process is repeated multiple times, as seen in GPT-2, which occurs approximately eleven times. For text generation, positional information of each word is embedded to preserve the sequential nature of language, ensuring that the model accounts for the order of words.

## Positional Encoding

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

In the context of natural language processing, each token is represented by a vector. For instance, 'food' and 'position' are mapped to distinct vectors. Similarly, 'position one' and 'position two' are associated with their respective vectors. By summing these vectors, we obtain a final embedding that encapsulates both the meaning and position of the tokens. This method maintains the dimensionality of the vectors, ensuring no alteration in their structure.

Alternatively, the original paper employs sinusoidal encoding, which generates vectors

## Computational complexity

The upcoming lectures will delve into the computational complexity of attention mechanisms in deep learning, particularly in the context of image processing. The attention matrix, with its substantial parameter count, raises questions about its necessity and efficiency. For each attention head, selecting 12 to 15 parameters increases the embedding space dimensionality. The subsequent multiplication by three for the query, key, and value matrices, combined with the number of attention heads, results in a significant parameter count. Adding dense layers further increases the parameter count, with normalization layers doubling the number of parameters. Despite the large parameter count, transformers demonstrate efficiency in large models, which is essential for processing complex language data.

## Transformer model structures

The paper's approach is bifurcated into an encoder and a decoder. The encoder processes input data into an abstract latent space, while the decoder translates this space back into the desired output, such as text generation or classification. Encoder models are prevalent in classification tasks like sentiment analysis, whereas modern models like GPT prioritize the decoder for text generation and sequence-to-sequence tasks.

## Encoder transformers

The next week's focus will be on decoder transformers, which are essential for text generation. These models, exemplified by GPT and Llama, predict the next token in a sequence, generating text by appending words. The input-output relationship in these models is a sequence of tokens with corresponding probabilities, which are used to generate text.

## Encoder-decoder models

The lecture will conclude at 12 PM, followed by a meeting on Monday morning, aiming for a smooth technical transition.
