# more about LLMs

The next three weeks will focus on deep learning applied to image inputs, covering computer vision tasks such as image classification, object detection with bounding boxes, and person counting.

The course will explore modern and relevant tips and tricks for large language models and their use in various applications beyond text and text generation.

A project week will follow Easter, allowing students to work on their projects, with an exam scheduled for May 12th.

The course will delve into optimizing large language models for today's hardware landscape, including techniques to reduce memory requirements for running these models on local hardware.

## Training decoder models

Large language models, such as those used in deep learning, typically predict the next token or word in a sequence. They do not create a structured plan for sentences but rather predict one token at a time, applying the attention mechanism to the input and the predicted token. This approach, while computationally intensive, is effective.

Emerging models diverge from this one-token-at-a-time prediction. These newer models are not yet widely adopted. Current users of ChatGPT engage in next-word prediction, which is straightforward to visualize and train. By masking the end of a sentence and providing the model with the next word, the model can predict and optimize parameters through loss computation. This training loop can be repeated across extensive text data, making the process efficient.

This training method, known as masked or causal attention,

## Generating text from a decoder model

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

For classification models, the last layer is a dense layer with the number of units equal to the size of the vocabulary, which is the number of tokens the model needs to learn. This can often be tens of thousands for models like GPT-2, which has around 40,000 words. Most state-of-the-art models typically have a vocabulary size of several hundred thousand. The simplest approach is to use the softmax activation, which outputs values between zero and one. If a token is very likely, its value will be close to one, while unlikely tokens will be close to zero. To generate text, this is run in an autoregressive loop, similar to predicting steps in time series data. You feed the prediction from the previous step as the next input, and continue this process. If you start with a short prompt, the generated text can be very long. This method works similarly to sampling from a distribution and selecting the most probable word.

In practice, the scores you get are influenced by the initial inputs and past predictions. These scores form a list of 40,000 elements, giving a score for each possible token. Typically, the scores for words that don’t make sense in context will drop quickly. However, the list remains long, containing all possible tokens.

If you always pick the most probable word, you quickly get stuck in a loop, generating the same short sequence repeatedly. This is because the model becomes deterministic, producing the same output each time. A better approach is to predict multiple steps ahead and select the most probable sentence. This involves evaluating the likelihood of different branches of text and choosing the one with the highest overall probability. This method, known as

## Softmax with temperature

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

In the context of language models, the softmax function is a crucial component in determining the probability distribution over a set of possible outcomes. The adjusted softmax, which includes a temperature parameter 't', modifies the distribution by controlling the randomness of the sampling process. A lower temperature results in a more deterministic output, favoring the most likely outcomes, while a higher temperature introduces more randomness, potentially leading to more creative and diverse results. The choice of temperature is application-specific and requires tuning to achieve the desired balance between formality and creativity.

## Large language models (LLMs)

Large language models, based on the transformer architecture, range from one billion to hundreds of billions of parameters. Models with seven to eight billion parameters are considered lightweight.

## Training LLMs

The next three weeks focus on training and fine-tuning deep learning models for specific tasks. The initial training involves self-supervised learning on large datasets, which forms a foundation model. However, due to cost constraints, training on the entire internet is impractical. Instead, models are fine-tuned for specific tasks, such as question answering, using datasets of questions and answers. Chatbots initially generate responses by predicting subsequent words, but fine-tuning improves their accuracy. Continuous fine-tuning and model merging are strategies for enhancing model performance.

## Training data

"To tune large language models for specific tasks, training data is essential, often sourced from high-quality internet resources such as Wikipedia, academic papers on arXiv, and Stack Overflow. For fine-tuning, annotated datasets with questions and answers are required. Large AI companies keep their training data confidential, making it difficult to obtain datasets comparable to those used by OpenAI. Reinforcement learning from human feedback is a costly method that involves human interaction to rate chatbot responses."

## Distributed training

The upcoming three weeks will delve into deep learning with a focus on image inputs. The course will cover the necessity of distributed training for large language models, which involves using data centers with multiple compute units. The LLaMA 3.1 model, requiring 3.3 terabytes of RAM per GPU, exemplifies the need for distributed training. Data can be split into batches for separate processing, and different layers can be trained on different GPUs. Inputs are distributed across computers, and a high-speed network is essential for aggregating weights. TensorFlow and Hugging Face are mentioned as tools for distributed training. The course will also explore using transformers for audio and image processing.
