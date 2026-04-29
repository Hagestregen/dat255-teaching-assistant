# training deep neural networks

The upcoming three weeks will delve into deep learning with a focus on image inputs. The curriculum will cover neural network fundamentals, tensor operations, and gradient descent optimization. A recap of machine learning objectives, such as underfitting and overfitting, will be provided. Students are expected to review the relevant chapters for a foundational understanding. Lab work will involve image classification tasks, including handwritten digit recognition and cat image analysis. The course will progress through various convolutional neural network concepts and training methodologies.

## Keras layers

The upcoming exercises will focus on convolution layers, which are pattern recognition layers that identify specific features in images using filters. These filters learn from example images to recognize patterns.

## Decomposition into simple patters

The next three weeks focus on deep learning applied to image inputs. The process involves using convolution layers to extract image features, pooling layers to downsample and focus on larger image parts, and dense layers to make final decisions based on identified patterns.

## The Conv2D layer

The upcoming sessions will explore convolutional neural networks (CNNs) for image processing. Key parameters include the number of filters and their size, with typical values ranging from tens to hundreds of pixels. Filters are usually kept small to maintain efficiency, but adjustments may be necessary for high-resolution images. Padding options, such as'same' padding, allow for output dimensions to match input dimensions by setting border pixels to zero.

## Pooling layers

The next three weeks focus on deep learning applied to image inputs. The pooling step in convolutional neural networks (CNNs) is crucial for downsampling images, which helps in recognizing objects regardless of their size within the image. Two common pooling methods are max pooling, which selects the largest activation value, and average pooling, which averages the values. Max pooling is typically preferred for image processing tasks.

## The MaxPooling2D layer

In deep learning applications, model size and padding are critical hyperparameters. Default input size is typically two by two, but three by three is also an option. Padding can be applied to individual pixels or in larger strides. The choice of using convolution and max pooling in multiple steps is also a consideration.

## The Dense layer

In the context of image classification, the final decision for a model to output a label such as "cat" involves a dense layer with a number of units corresponding to the number of classes. The choice of units and activation function are critical components that will be discussed in detail.

## My first convolutional network

The next three weeks will focus on deep learning applied to image inputs, specifically in the realm of computer vision tasks such as image classification, object detection with bounding boxes, and person counting.

In the context of neural networks, the sequential model is a straightforward approach where layers are added one after another. The first layer requires input shape specification, which for grayscale images like MNIST, is 28x28 pixels. The sequential model automatically calculates the necessary connections, weights, and biases for each layer based on the input shape.

The process begins with defining the input layer, followed by stacking various layers such as convolutional layers, which extract features from the image by analyzing small patches, and max pooling layers, which reduce the spatial dimensions of the feature maps. This combination allows for the extraction of both local and more global features.

After several convolutional and pooling layers, the output is flattened into a one-dimensional vector, which is then passed through a dense layer to perform classification. The dense layer's output size corresponds to the number of classes in the classification task.

The activation function, which introduces non-linearity into the model, is crucial for enabling the network to learn complex patterns. The summary of the model provides an overview of the architecture, including the number of parameters in each layer, which is automatically computed by Keras.

This approach to building a convolutional neural network (convnet) is efficient as it abstracts away the complexity of parameter management and layer connections, allowing for rapid prototyping and experimentation. The upcoming three weeks will delve into deep learning with a focus on image inputs. The curriculum will cover computer vision tasks such as image classification, object detection with bounding boxes, and person counting.

In the context of convolutional neural networks (CNNs), the transition from color to grayscale images results in a 26x26 pixel representation. Each filter activation is considered as a channel, leading to a 32-channel image. Max pooling reduces the spatial dimensions by half, maintaining the filter activations. Subsequent convolutions further reduce the image size, eventually condensing the activations into a 800-value vector for classification.

Model compilation involves setting up the architecture and selecting an appropriate loss function. For classification tasks, cross-entropy is commonly used. Optimization techniques will be discussed in the following week. Metrics such as accuracy, recall, and precision can be added to evaluate model performance.

Data preparation for training involves selecting input features (x) and corresponding labels (y). The choice of batch size and the number of epochs for training are crucial considerations. Stochastic gradient descent, which uses random samples, is the default optimization method. Data shuffling can be enabled to ensure variability in the training process.

The `model.fit` method initiates the training process, which may include a validation set for monitoring performance. Training typically involves multiple epochs to refine the model, with the expectation that the loss decreases over time.

## Decomposition into simple patters: Theory vs practice

The curriculum for the next three weeks will delve into deep learning techniques for image inputs, with a focus on computer vision tasks such as image classification, object detection with bounding boxes, and person counting. The approach involves decomposing complex patterns into simpler components and reconstructing them through a multi-level hierarchy to facilitate decision-making processes. This methodology is exemplified by the use of convolutional neural networks (CNNs), which are adept at recognizing patterns in images. A practical demonstration involves testing a CNN trained to identify cats on an image not present in the training dataset, thereby assessing the efficacy of the model's learned hierarchical pattern recognition.

## Layer activations

The next three weeks will focus on deep learning applied to image inputs, specifically in the realm of computer vision tasks such as image classification, object detection with bounding boxes, and person counting.

In the exploration of convolutional neural networks (CNNs), we observe that the initial layer filters are designed to identify simple shapes and patterns. For instance, one filter may highlight diagonal or curved lines, while another may act as a white color recognizer. These filters, although seemingly random, are crucial in the early stages of the model's learning process.

As we progress to the second layer, the resolution decreases due to max pooling, which retains high-value pixels and discards less significant ones. This reduction in resolution allows the model to focus on higher-level patterns, such as detecting whiskers or facial features like eyes and noses.

By the third layer, the patterns identified by the filters become less distinct to human observers but are essential for the model to recognize more complex structures. The fourth layer further reduces resolution, emphasizing the most relevant features, which, in the context of cat images, could be the unique identifiers or "fingerprints" of the subject.

The presence of filters with no activations does not necessarily indicate an excess of filters in that layer. Instead, it signifies that the filter did not detect any relevant features. Conversely, filters with activations indicate the presence of features that the model has learned to recognize. These activations are instrumental in distinguishing between images containing cats and those that do not, ultimately guiding the decision-making process in image classification tasks. The desired behavior in convolutional neural networks (CNNs) for image classification is for most filters to activate, indicating the presence of a feature, and fewer filters to activate, suggesting a specific class. This pattern helps in identifying the class, such as recognizing a cat when most filters activate.

## What is a Convolutional Neural Network?

The next three weeks focus on deep learning applied to image inputs, with a particular emphasis on computer vision tasks such as image classification, object detection with bounding boxes, and person counting.

In the interactive visualization tool, users can apply various filters to images, observing the effects of each filter. The tool demonstrates how filters shift across positions and how merging colors results in distinct outputs. The visualization also highlights the transformation of outputs by activation functions, which are non-linear functions that make the outputs more meaningful. Users can explore these values and their contribution to the next convolutional layer. The system's ability to correctly identify objects, such as a sports car, underscores the significance of analyzing the activations.

Convolutional layers are a key component of deep learning models, offering

## The activation function

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

Convolutions were initially difficult functions, but now let's look at the simplest types of functions. In neural networks, an essential component is the activation function, which can appear in various forms. I claimed earlier that the key aspect is that it’s not a straight line; if it were, we could argue that adding more layers would result in a linear composition, making the overall model linear. While this argument holds, the point remains valid: to achieve more flexibility and complexity, we need to introduce non-linearity through the activation function.

There are several considerations for a good activation function. First, it must be non-linear. Second, for gradient descent to work, the function must be differentiable. As long as we can compute derivatives for…

## Last layer activation functions

The next three weeks focus on deep learning applied to image inputs. Classification tasks require different model outputs compared to regression tasks. After feature extraction and pattern identification, the model must decide on the output: classification or regression. Classification involves assigning a label (e.g., 'cat' or 'dog') or a score between zero and one, while regression predicts a continuous value (e.g., house prices). For classification, softmax activation is used for multi-class scenarios to ensure the sum of scores equals one, representing probabilities. In Keras, sigmoid activation is used for binary classification, and softmax for multi-class problems.

## Finding optimal parameters

Deep learning models, particularly those with multiple dense layers, involve a complex optimization process due to the interconnected nature of the layers and the vast number of parameters. The optimization challenge was overcome in the early 1990s with the development of backpropagation, a method that systematically adjusts all parameters in the network. This technique is central to training deep learning models and is crucial for understanding their functionality.

## Backpropagation

The next three weeks focus on deep learning applied to image inputs. The process involves defining layers and nodes, initializing parameters randomly, and running examples through the model to obtain predictions. These predictions are compared to the targets using a loss function that guides the optimization of parameters. The optimization involves a forward pass to compute predictions and a backward pass to compute gradients, which are used to adjust the parameters iteratively.

## Gradient descent

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

In deep learning, optimizing model parameters is crucial for minimizing loss. Starting from a random point, the gradient of the loss with respect to parameters indicates the direction of steepest descent. By iteratively moving in the negative gradient direction, the model converges to a point of minimum loss, representing the best model parameters. This process, known as gradient descent, involves a hyperparameter called the learning rate, which determines the step size. Choosing an appropriate learning rate is essential for efficient training, as too large steps may overshoot the minimum, while too small steps can slow down convergence. The updated parameters are calculated by adding the gradient change to the current parameters. This fundamental technique is central to training deep learning models.

## Learning rate

The art of machine learning lies in balancing the pace of progress, avoiding both slow advancement and overshooting the target. Model evaluation through metrics helps identify and rectify such issues. Next week, experiences with this technique will be shared in notebooks. Despite the ease of documenting network architecture and code with modern libraries, practical challenges persist, particularly in navigating loss landscapes with multiple minima, which can hinder optimal performance.

## Practical problems

"Deep learning models often experience plateaus where progress stalls before sudden improvements. This can be addressed by adjusting learning rates or optimization methods, or by restarting the training process. The phenomenon is typical in deep learning, where models may appear to be nearing optimal solutions but temporarily halt in progress. No error messages are present, suggesting the model's parameters are correct, yet the performance is not ideal. Experience with these challenges will be developed in the following week's session."

## Vanishing and exploding gradients

Gradients guide the optimization of models in deep learning. They indicate the direction of steepest ascent, with the goal of minimizing loss. However, issues like vanishing and exploding gradients can impede learning. Vanishing gradients lead to negligible updates, while exploding gradients cause instability. Analogous to a chain of microphones and amplifiers, each layer's stability is crucial. Techniques to maintain parameter stability include input normalization and careful model design.

## Some tricks

The next week's focus will be on activation functions, particularly the challenges posed by saturating functions like the S-shaped function, and the benefits of using ReLU. Initialization strategies for neural network parameters will also be discussed, with Keras's default settings as a starting point. Layer definitions will be covered, emphasizing the importance of choosing appropriate activation functions and initializers. Normalization layers will be introduced as a technique to improve model performance.

## Regularisation

The lecture will cover the evolution of regularization techniques, focusing on the transition from traditional \( L_1 \) and \( L_2 \) regularization methods to newer approaches. The \( L_1 \) and \( L_2 \) regularization methods aim to reduce overfitting by penalizing large parameter values, with \( L_1 \) summing the absolute values and \( L_2 \) summing the squares of the parameters. Despite their simplicity and ease of implementation, these methods are considered outdated. The lecture will introduce more advanced regularization techniques that address the limitations of \( L_1 \) and \( L_2 \) regularization.

## Dropout regularisation

Dropout is a regularization technique in deep learning that involves randomly setting a subset of activations to zero during training. This prevents co-adaptation of nodes and encourages the network to learn more robust features. During inference, all nodes are used, but dropout is not applied. It is typically set to a percentage between 10% and 50%.

## Putting together an improved network

The lecture covers the application of deep learning to image inputs, focusing on the process of refining neural networks. The initial step involves normalizing image pixel values from the range of 0-255 to a 0-1 scale, enhancing the data's standardization. Following this, convolutional layers are applied, followed by activation functions such as ReLU. Normalization techniques are introduced to stabilize training and prevent overfitting. The network architecture is progressively built by adding layers and optimizing parameters through learning. The final model maintains a consistent number of parameters, demonstrating a more sophisticated network design.

## Best practices

"Deep learning requires careful consideration of layer configurations, activation functions, and parameter initializers. While guidelines are provided to streamline the process, they are subject to change as the field evolves. Adaptability to new best practices is essential for efficiency in problem-solving."
