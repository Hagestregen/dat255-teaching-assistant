# natural language processing: embeddings

The upcoming two weeks will delve into large language models, with a focus on transformers that power these models. The first week will cover the foundational concepts, while the second week will explore their applications beyond the scope of the current textbook. Lab notebooks will include exercises on text model embeddings, a key concept in understanding text models.

## From yesterday:

The course emphasizes enhancing numerical representations for machine learning models. Techniques for transforming data into meaningful numerical representations are explored, with a focus on improving these representations for better decision-making.

## Embeddings

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

Embeddings represent categorical values as vectors. In text processing, categorizing tokens as words and encoding them into embeddings involves initializing random numbers for each token, which are then optimized during training. The dimensionality of the embedding space affects the training data requirement and the flexibility of the space. Embeddings learn meaningful vector representations for words, positioning them in a vector space that facilitates visualizations and distance calculations. To map words to vectors, one-hot encoding is followed by a dense layer using the `Keras.layers.embedding` layer. The size of the vocabulary and the dimensionality of the embedding space are key considerations, with modern

## TensorBoard

The next three weeks focus on deep learning applied to image inputs, with a particular emphasis on computer vision tasks such as image classification, object detection with bounding boxes, and person counting.

The project week is centered around TensorBoard, a tool that facilitates the visualization of learning curves and various other metrics. It also includes a projector tool, which allows for the visualization of vector spaces. An example of this is the visualization of the embedding space for GPT version 2, a model that fits on a laptop and is open-sourced. The embedding space is 768-dimensional, and tools exist to project these dimensions down to three dimensions for visualization purposes.

The training of a large language model involves converting words into numbers using embeddings, which are then fed into a transformer. The projector tool visualizes the embedding space, which is crucial for understanding the semantic relationships between words. For instance, words like "open," "public," and "free" are close in meaning and are positioned similarly in the embedding space. This demonstrates the concept of statistical semantics, where the distance between points in the embedding space correlates with the meaning of words.

The GPT-2 model, released in 2017, serves as an example of how embedding spaces are created and utilized. The model was trained on data without explicit explanations, allowing it to infer relationships among words based on context. The embedding space preserves the semantic meaning of words, with closely related words placed in similar areas. This approach enables the creation of a dictionary of similar words through training on data.

The distances between points in the embedding space carry meaning, reflecting the semantic relationships between words. This is demonstrated by the movement of tokens like "ocean" and "bay" in the embedding space, which move to positions that reflect their semantic meanings during training. The model demonstrates its ability to extract meaningful semantics from both textual and numerical data.

## Computing similarity in embedding space

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

To compute the similarity between words, we can use two measures: Euclidean distance and cosine similarity. Euclidean distance calculates the straight-line distance between two points in a multi-dimensional space, while cosine similarity measures the cosine of the angle between two vectors, indicating the directional similarity regardless of their magnitude.

Euclidean distance is computed by taking the difference between corresponding elements of two vectors, squaring these differences, summing them up, and then taking the square root of the sum. This metric is intuitive and scales linearly with the dimensionality of the space.

Cosine similarity, on the other hand, calculates the cosine of the angle between two vectors. It is derived from the dot product of the vectors divided by the product of their magnitudes. Cosine similarity is particularly useful when the magnitude of the vectors is not of interest, and we are more concerned with the direction they point in.

In the context of deep learning, these metrics can be used to compare embeddings of words or images, aiding in tasks such as finding similar items or grouping them based on their features. The next three weeks will focus on deep learning applied to image inputs, with tasks including image classification, object detection with bounding boxes, and person counting.

In the context of embedding spaces, the Euclidean distance may not always be the most effective measure for similarity, as it does not account for the directionality of vectors. Instead, cosine similarity, which considers the angle between vectors, can often provide a more meaningful assessment of similarity.

Words with similar meanings tend to cluster together in embedding spaces, which can be leveraged when training language models to select semantically relevant words.

Projects related to the course are due by April 30th, with no extensions allowed. Starting work early is recommended to ensure a smooth progression towards the deadline.

Embeddings involve vector spaces where mathematical operations such as subtraction and addition can be performed, resulting in vectors that maintain consistency. For example, comparing "man" to "woman" in an embedding space yields a vector that represents a gender dimension.

The relationship between a country and its capital can be visualized in an embedding space, where moving from a country to its capital often results in a vector pointing in the same direction, indicating a consistent relationship.

Vector arithmetic in embedding spaces can lead to sensible results, aiding in the recognition of patterns and the interpretation of distances and directions.

While embedding spaces can provide valuable insights, not all tokens may align with human understanding, and the model's representation of the world may not always conform to our expectations.

## Better text tokenisation

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

Tokenization simplifies language processing by grouping words into lemmas, reducing vocabulary size and complexity. This process, while useful, is often manual and requires domain expertise, especially with limited data. Sub-word tokenization, a method used in natural language processing, splits words into smaller units, which can be combined to form words. This technique varies across models and requires the correct tokenizer for proper preprocessing. In computer vision, similar preprocessing steps are necessary for model training.

## The Tokenizer Playground

"For tokenizing text, model selection dictates specific IDs. Consistency in starting numbers is crucial for pre-training and using pre-trained models."

## Pretrained embeddings and models

The next three weeks will focus on deep learning applied to image inputs, with an emphasis on computer vision tasks such as image classification, object detection with bounding boxes, and person counting. For image processing, the use of pre-trained models is recommended due to their efficiency and effectiveness. These models, which can be fine-tuned for specific needs, start with simple pattern recognition layers that can be progressively enhanced. Fine-tuning pre-trained models is a highly efficient approach, as it often yields similar results to training from scratch. In the context of language models, embeddings created from large datasets provide a general vocabulary that can be fine-tuned for specific applications. Embeddings can be obtained from various sources, including open-source models or purchased from providers like JetGPT.

## N LP resources

TensorFlow provides text preprocessing tools, including normalization and subword tokenizers, which are useful for language modeling tasks. Hugging Face offers a comprehensive repository of pre-trained language models.

## Hugging Face

Hugging Face provides a platform for downloading and testing various models, including DeepSeq and Quan, with guides available for local execution. Users can utilize Colab, other providers, or specific libraries for running these models. Google's Gemma and OpenAI's GPT-OSS are also accessible for experimentation. Running large language models locally is a feasible project for those interested in creating chatbots or comparing models without a license.

## Other uses of embeddings

Embeddings, while commonly associated with word representations, can also be applied to other entities. Google's search results often reflect this concept, as it uses Vector Search to find similar content rather than exact keyword matches. This method involves computing vectors for search queries and ranking results by their similarity, effectively placing items in a vector space to identify similarities. This technique is crucial for large language models and is a familiar experience for users.

## Embed everything

The upcoming lectures will explore multimodal deep learning models capable of processing and generating content across various data formats, including text, images, and audio. These models utilize embeddings to map different types of inputs into a shared high-dimensional space, enabling the generation of related outputs, such as images from textual descriptions.

## Sentence embedding

The Universal Sentence Encoder creates embeddings for full sentences, allowing for semantic comparison using cosine or Euclidean distance. It can discern semantic similarity between questions like "How old are you?" and "What is your age?" but not between "Your cell phone looks great" and "How old are you?" This demonstrates the encoder's ability to represent related concepts in proximity within an embedding space.

## Image embedding

The next three weeks will explore deep learning with a focus on image inputs. We will delve into embedding spaces, where images are represented as vectors, and discuss the challenges and methods of placing images into these spaces. We will also examine the use of convolutional neural networks (CNNs) for pattern extraction and classification, as well as the potential for generating images from embeddings. The exploration will include the concept of image similarity and the ability to perform similarity operations beyond traditional classification tasks.

## Audio embedding

The upcoming week introduces advanced techniques for processing sequence data, including speech, using tools like Whisper for transcription. Embeddings based on similarity are created for further processing, with convolutions and LSTM models being effective for sequence data. New modeling strategies will be discussed, emphasizing the computational intensity of transformers and the importance of embeddings in both language and speech models.

## Latent spaces

Latent spaces are abstract representations where improved decision-making numbers reside within a deep learning model. These spaces, often referred to as latent feature spaces, are obtained automatically for each layer in a model with multiple layers. The progression of these spaces is such that they become more specific and better as the model processes information further. Transfer learning is employed to utilize pre-trained models, which allows for the reuse of general latent spaces for specific tasks, reducing the need for extensive training data and preventing overfitting. This method is especially beneficial when dealing with limited data or computational resources, as it leverages the knowledge from models pre-trained on large datasets.

## Pretrained embeddings

For the project, training from the beginning is required. However, pre-trained embeddings are available for use. A leaderboard displays the performance of embeddings on various tasks. Regularly check the leaderboard to select an effective and hardware-compatible embedding. The updated list includes performance metrics and a plot, which details the embeddings' effectiveness. Pre-trained embeddings can be downloaded and selected based on project requirements.

## Measuring embedding quality

"Embedding quality is assessed through textual similarity tasks, where human judgment on sentence similarity is used. Model performance is evaluated by comparing generated vector similarities to expected values. A successful model aligns closely with semantically related sentences, while misalignment with unrelated sentences indicates poor performance."

## Contextualised word embeddings

The upcoming week's focus will be on addressing polysemy in word embeddings, determining word meanings based on context, and selecting the appropriate semantic interpretation.
