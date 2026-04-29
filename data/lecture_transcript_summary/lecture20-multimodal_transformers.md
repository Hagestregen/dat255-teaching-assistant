# multimodal transformers

The upcoming lectures will explore the versatility of transformer networks, initially designed for text and text generation, and their application to various data types. The adaptability of transformers necessitates careful consideration of tokenization when applied to non-text data.

## Training LLMs

The lecture covered the concept of self-supervised pre-training in the context of Generalized Pre-trained Transformers (GPT), where 'P' denotes pre-training. It also addressed fine-tuning, a process that enhances the model's utility for specific tasks. Distributed training was mentioned as a method for handling large datasets.

## Distributed training

"In the context of deploying large models, engineering for hardware compatibility is essential for ambitious projects. For our purposes, a single-machine fit suffices. Large language models require distribution across multiple compute nodes due to their size."

## Supervised fine-tuning

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

Supervised fine-tuning is introduced to address the limitations of autoregressive training in question-and-answer settings. It involves training a model with actual solutions to instructions, which is a costly process in terms of data creation. Data can be sourced from platforms like Stack Overflow, but curating high-quality solutions is labor-intensive. Training a language model requires data scraping, self-supervised pre-training, and supervised fine-tuning. High-quality data is crucial for creating effective chatbots, and major AI companies invest in collecting such datasets. Pre-trained models can be adapted for specific tasks using "instruct," "adapters," or "fine-tunes."

## LLM fine-tuning with limited resources

Fine-tuning deep learning models typically demands extensive computational resources, which are often beyond the reach of smaller projects. Despite this, it is possible to achieve comparable results with more constrained resources. One strategy involves using a fully pre-trained model and retraining it from scratch, which is resource-intensive due to the substantial memory requirements, including for the optimizer. A significant challenge in this process is catastrophic forgetting, where the model may lose previously acquired knowledge during fine-tuning.

To mitigate this, targeted fine-tuning methods are employed. Prompt tuning is one such method, where a smaller model is integrated with a large language model to customize the embedding space

## Low-rank adaptation (LoRA)

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

In deep learning models, the input data is represented as matrices. The size of these matrices is determined by the number of tokens (\( n \)) and the dimensionality of the embedding space (\( d \)). The sequence length can vary significantly, from a few hundred to tens of thousands or even millions of tokens. These matrices can be large, but they must be managed within the hardware constraints.

To manage the size of the weight matrix in a transformer block, small matrices are added. These matrices are used for matrix multiplication, resulting in a matrix of size \( n \times d \). A new parameter \( r \) is introduced, which determines the number of new parameters added to the model. The choice of \( r \) depends on the available computational resources; a larger \( r \) requires more GPU memory, while a smaller \( r \) reduces memory usage.

The optimizer, which tracks the history of parameters, requires significant memory. However, only the parameters of the smaller matrices being trained need to be tracked. The pre-trained model's parameters remain unchanged, and only the additional parameters introduced by the…

## Quantization

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

In machine learning, numerical precision can be optimized by reducing the number of bits required for each number. This allows for the use of fewer bits for weights and activations in each layer, such as using 16-bit floats instead of 32-bit floats. Specialized systems like drones may use 8-bit integers to save space. Model names from Hugging Face often include quantization levels, such as "q4" for 4-bit quantization. Downloading models with options like "quantized to 8 bits" can

## Knowledge distillation

Knowledge distillation involves training a smaller model to replicate the behavior of a larger model. The larger model, acting as a teacher, provides answers that the smaller model learns to mimic. This technique is applicable to tasks such as classification and reasoning.

The process includes using a large language model to generate answers, which are then used to train the smaller model. This method allows the smaller model to learn without storing the entire set of parameters from the larger model.

Examples of knowledge distillation can be found on platforms like Hugging Face. However, companies like Claude have faced legal challenges for using this technique to transfer knowledge from proprietary models to open-source ones.

Training for knowledge distillation involves sending questions to the large model, receiving answers, and using those answers to train the smaller model. This approach is

## Transformers for other data types

The upcoming sessions will explore multimodal transformers capable of processing diverse data types, including text and images, and generating corresponding outputs. This presentation aims to expand understanding of transformer applications, linking them to core course concepts.

## General transformers

The upcoming three weeks will delve into deep learning for image inputs, exploring computer vision tasks such as image classification, object detection with bounding boxes, and person counting.

Transformers, originally designed for text processing, are versatile enough to handle sequences of various types, including images, videos, and audio. Unlike convolutional networks that rely on local pixel or time point relationships, transformers use a large attention matrix to consider the entire sequence simultaneously, enabling them to capture long-range dependencies. This approach is applicable to a wide range of data types, including point clouds, which are commonly used in self-driving cars for 3D spatial representation.

In the context of deep learning, different types of input data are referred to as modalities. A multimodal model can process and integrate information from these various modalities, allowing for comprehensive analysis and interaction with diverse data forms.

## Visual attention

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting. The concept of attention in deep learning models helps to identify relevant areas of an image for tasks such as caption generation. By analyzing attention matrices, we can see which parts of an image are associated with specific words in the generated captions. For example, areas around objects like frisbees or people are highlighted when those objects are queried. This mechanism, while established, can be applied to various tasks beyond images, including other data types.

Transformer-type models that process images and generate captions can be analyzed using attention matrices to understand the model's focus. For instance, attention scores for a specific object, like a giraffe, may highlight the background rather than the object itself. This can help identify errors in the model's interpretation, such as confusing a skateboard for a violin. By examining attention scores, we can debug and improve deep learning models, ensuring they generate sensible outputs. The attention mechanism also allows for the exploration of relationships between different concepts, both in text and…

## Transformers for computer vision

The complexity of processing images in deep learning arises from the need to extract meaningful information from pixels, which requires contextual understanding. Unlike text, where words serve as discrete units of meaning, images necessitate a more nuanced approach. The simplest method, treating each pixel as a token, fails to capture the essence of an image. Instead, a matrix of neighboring pixels is analyzed, resulting in a quadratic number of tokens relative to the image dimensions. This approach is manageable for text but challenging for images due to the sheer volume of tokens. To address this, techniques such as dividing the image into patches or using convolutions and downsampling have been developed. These methods effectively reduce the token count

## Vision transformer (ViT) for classification

The vision transformer (VIT) architecture, a general-purpose computer vision model based on transformers, processes images by converting them into patches, which are then embedded into a sequence of tokens. This method retains the original image's concepts through patching. Positional embeddings are used to maintain the spatial relationships between patches. The sequence of embedded patches, along with class information if necessary, serves as input to the transformer, enabling it to perform various tasks.

## Positional embeddings

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting. Positional embeddings are used in transformer models, with learned embeddings being the standard approach. These embeddings can be extended to any number of dimensions, making them suitable for various data types, including medical imaging.

## Combined architecture

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

The transformer architecture is versatile, capable of handling various tasks. Preprocessing and task-specific work are necessary to prepare data and extract or predict information. Training such models requires substantial data due to their generality.

Illustrative examples include using transformers combined with other methods for tasks like identifying seagulls in images. Previous architectures, such as convolutional networks and recurrent networks, are useful for initial feature extraction. A CNN can transform pixel data into a sequence for transformer input.

Positional encodings are essential for the transformer's function, which can produce multiple outputs for different tasks. For example, one output might identify a single target, while another might identify a different target, and some outputs might indicate no target found.

In essence, a large transformer can be central with data-specific and task-specific components

## Inductive biases

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

Inductive bias is an assumption made when building a model. For example, using a linear model to predict house prices based on square meters assumes a linear relationship. This simplicity can lead to poor performance if the actual relationship is non-linear. However, it simplifies the training process by requiring less data.

Convolutional neural networks (CNNs) assume images are composed of recognizable patterns formed by pixels or neighboring pixels. They also assume translational equivalence, meaning the structure of an image remains consistent regardless of its position.

Recurrent neural networks (RNNs) used for time series assume a sequential order among neighboring time points, with the most recent point being closer to the end. This assumption is beneficial for understanding temporal sequences.

Transformers assume a relational structure among data points, allowing them to handle tasks without requiring data to be ordered or follow a specific shape. This generality makes them versatile but requires a large amount

## Combining vision and text

The upcoming lectures will explore the versatility of deep learning models in performing multiple tasks, such as object detection, classification, and segmentation in images. These models can identify objects, determine their locations, and classify them, showcasing their ability to handle various specific tasks with a single general model.

## Audio transformers

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

In the context of audio processing, speech-to-text technology is a notable application. For instance, during a meeting, one can transcribe spoken words using speech-to-text services. Chachipity is an example of a service that facilitates interactive conversations or generates text-to-speech responses. Text-to-music generation is another intriguing application, where services can create music based on textual prompts.

The Whisper model, an open-source speech-to-text tool, operates by converting spectrograms into text. It is based on the transformer architecture, which incorporates positional encodings and transformer blocks. The attention mechanism is central to its function, with self-attention examining the influence of words on each other and cross-attention relating sequence inputs to…

## The Challenge of StarCraft

Transformer models are versatile, capable of playing games and performing a wide range of tasks by tokenizing and processing diverse data types.

## Multimodal transformers

The upcoming three weeks will delve into deep learning with a focus on image inputs. The course will explore computer vision tasks such as image classification, object detection with bounding boxes, and person counting. The curriculum will progress from general pattern recognition to the creation of images, emphasizing the importance of tokenizing and embedding data for operation in latent spaces. This approach, rooted in the extraction of patterns and subsequent pattern recognition in vector spaces, will be revisited with a focus on generative tasks. The course will culminate in a project week, where students will apply their knowledge to develop projects, potentially including web app deployments and ML operations.
