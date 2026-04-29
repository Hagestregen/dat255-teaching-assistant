# augmentation and advanced computer vision

The upcoming sessions will delve into advanced functionalities of the Keras library, with a focus on flexible neural network assembly methods and computer vision techniques. The emphasis will be on data augmentation strategies to enhance training datasets, illustrated with examples of synthetic image data. The discussions will also touch upon the creation of neural network models for image inputs, including object detection and image classification tasks.

## Improving generalisation

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

In machine learning projects, especially those involving deep learning, the availability of training data is often limited. To improve generalizability, one can artificially modify existing examples. This involves creating variations of the data, such as rotations or inversions, which maintain essential features while expanding the diversity of the dataset. This technique enhances the robustness of machine learning models by effectively increasing the volume and variety of training data.

## Augmentation

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting. Data augmentation is a technique used to enhance the diversity of training data by applying random transformations to images, such as flipping, rotating, cropping, adjusting brightness, changing colors, and applying equalization. During training, these transformations are enabled to generate diverse data, improving model performance. For inference, the augmentation layer is set to `training=False` to ensure accurate predictions. Effective augmentation methods include `RandomFlip`, `RandomRotation`, `RandomCrop`, `RandomBrightness`, `RandomColorJitter`, and `Equalize`. The choice of transformations depends on the task and dataset. Studies indicate that augmentation significantly improves model performance. To evaluate augmentation's effectiveness, compare test set accuracy before and after applying augmentation.

## More advanced network configurations

More complex models require extensive training data to accommodate the added flexibility. While data augmentation is beneficial, a substantial dataset is equally crucial. The previous week's discussion centered on the sequential model, known as Keras, which is utilized via `keras.Sequential`.

## Baseline model

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

In neural network design, layers are sequentially connected, with the output of one layer feeding directly into the next. This sequential connection is crucial as it allows for the propagation of information through the network. However, it also imposes a limitation: outputs from one layer cannot be directly connected to non-sequential layers.

This limitation raises two questions: Why would one want to bypass the sequential connection? And how can this be achieved? These questions lead to exploring advanced neural network architectures. The current state of the art in computer vision

## The Keras functional API

The next three weeks focus on deep learning applied to image inputs, with tasks including image classification, object detection with bounding boxes, and person counting.

In deep learning, sequential data processing is common, where transformations improve data representation iteratively. Initially, data is unsuitable for decision-making, but through extraction and transformation, it becomes a usable list of numbers for decisions. A dense layer then finalizes the decision-making process.

Alternatively, the functional API in Keras allows for more flexible layer connections. Instead of a sequential list, layers are defined as functions, with inputs and outputs explicitly named. This approach enables the creation of complex models with interconnected layers, forming a graph of connections. The model is assembled using these layers, and standard training methods like `fit` and `compile` are applied.

This method provides access to individual layers, allowing for customized manipulation and understanding of the model's internal workings.

## Example: Bird classifier

The lecture focuses on the integration of image and numerical data in deep learning models. The initial setup involves two distinct inputs: an image and numerical data. Image processing includes convolutions and feature flattening, while numerical data is processed through dense layers for pattern recognition. The features from both inputs are concatenated and further processed. The output is a score determined by a dense layer.

The computational graph is defined with inputs and outputs, where the inputs consist of the image and numerical data, and the output is a class score. This architecture supports multitasking, such as simultaneous bird classification and other predictions, with the potential to refine image-based classification using numerical data.

The combination of image and numerical data can enhance the model's performance, especially if there's a correlation between the two. The optimal point for combining these inputs may vary, and experimentation is key to determining the best approach.

Normalization layers are recommended to ensure that the numbers from different inputs are on a similar scale, preventing larger numbers from overshadowing smaller ones. This normalization is crucial for maintaining reasonable numbers throughout the model's layers.

The lecture also emphasizes the importance of fine-tuning to achieve reasonable numbers in the model's intermediate layers, which is essential for the model's effectiveness.

## Non-sequential networks

The next segment will explore the flexibility of neural network architectures, particularly in computer vision, by examining the modularity of layers and the potential for creating complex patterns through various connections. The discussion will include the use of loops and the selection of optimal network configurations based on research findings.

## Residual connections

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

Deep learning models consist of stacked layers, each performing a composite function on the data. This can lead to issues such as exploding or vanishing gradients. Residual connections, or skip connections, address these problems by reintroducing the input at various points in the network.

Residual connections involve copying the input and adding it to the output of each layer. This ensures that information flows through the network effectively, preventing the gradients from exploding or vanishing. This technique is particularly useful for very deep networks, where adding more layers beyond a certain point does not significantly improve performance.

In Keras's functional API, residual connections can be implemented by specifying two inputs for each layer: the transformed data from the previous layer and the original input. This allows for the creation of convolutional blocks and the application of sequential operations, such as max pooling, while ensuring that the shapes of the inputs and outputs match. The final output can be obtained by adding the results of the previous layers using an "add" layer.

Residual connections simplify the creation of complex architectures by providing a structured way to combine transformations. By visualizing the connections and defining the computational flow, it is possible to implement deep learning models efficiently.

## Residual networks

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

In neural networks, a sequential model with multiple layers can face challenges. If a layer has zero gradients, it becomes untrainable, halting backpropagation. To address this, residual connections are introduced, allowing gradients to flow through the network even when certain layers are untrainable. This technique enables the training of very deep networks, improving performance in computer vision tasks.

## Densely connected convolutional networks

The next three weeks will explore deep learning architectures, focusing on ResNet and DenseNet. ResNet employs residual connections to facilitate the training of deep networks, while DenseNet uses dense connections to improve efficiency. Both architectures utilize blocks of layers, including convolution, max pooling, and normalization, with batch normalization aiding in stable learning. Sideways connections form stacks, and a final dense layer is used for classification.

## Inception networks

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

Inception networks are a deep learning architecture that utilizes convolutions with varying filter sizes to capture different patterns. These networks consist of mini networks that are composed into larger networks. The inception module uses convolutions of different widths and dimensions to distinguish patterns, with the ability to concatenate these within a block.

GoogleNet is a notable example of an architecture that incorporates inception modules. It features multiple outputs from different inception blocks, which are differentiable functions that allow for learning from a large number of parameters. An extreme inception network explores various connections and adjustments to compute different outputs, treating RGB image channels separately for efficiency.

For projects involving multi-channel images, such as satellite imagery, the inception network architecture can be particularly beneficial due to the significance of different spectral ranges in the channels.

## Beyond CNNs: Vision transformers

The upcoming weeks will delve into the application of transformers to image processing. Transformers, initially developed for text, can be adapted for image tasks by segmenting images into parts and treating them similarly to words in a sentence. This approach has shown promise, particularly in the early 2020s.

Despite their potential, convolutional neural networks (CNNs) are currently the preferred method for image recognition due to their proficiency in pattern detection within images. CNNs are especially adept at handling complex image classification tasks.

As the course progresses, the use of transformers for image generation will be explored. For image classification and related tasks, convolutional networks are the standard choice. Techniques involving loops and…

## keras.applications

In the upcoming lectures, the focus will be on implementing deep learning models using Keras for image-based tasks.

## Feature extraction

The next three weeks will explore deep learning for image inputs, focusing on computer vision tasks such as image classification, object detection with bounding boxes, and person counting.


In the upcoming sessions, we will delve into the use of pre-trained models for image recognition tasks. These models, which are trained on extensive datasets like ImageNet, can be adapted to recognize a variety of objects, including animals, vehicles, and other common items. The process involves modifying the model to suit the specific categories present in your dataset, which may differ from the original training data. This adaptation can be achieved by replacing the classification layer with a new one tailored to the unique requirements of the dataset.


The approach allows for the retention of the model's inher

## Fine-tuning

The next three weeks focus on deep learning applied to image inputs, with tasks including image classification, object detection with bounding boxes, and person counting.

To adapt a pre-trained model for new tasks, one can add a custom layer on top of the base model and freeze the base model's parameters. After training the custom layer, the base model's parameters can be unfrozen and fine-tuned. Transfer learning is facilitated by using pre-trained models for feature extraction or fine-tuning.

The applications section offers a variety of pre-trained models, such as VGG, ResNet, Inception, DenseNet, and Convex, which are suitable for different considerations like device constraints and prediction speed.

For example, to use ResNet50 for feature extraction without the top classification layer, one can import the model with `resnet = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)`. To fine-tune the model for a specific number of classes, set `include_top=True` and specify the number of output classes. Freezing the base model's layers is done by setting `trainable=False`, and they can be unfrozen later for further training.

## Modern convolutional networks for computer vision

The notebook for this week should function as intended. If issues arise, assistance will be provided. The `plot_model` function is sought to visualize model architecture and corresponding shapes.

## Other computer vision tasks (next week)

The upcoming three weeks will delve into deep learning for image inputs, with a focus on computer vision tasks such as image classification, object detection with bounding boxes, and person counting. The following week will introduce segmentation, aiming to delineate the contours of objects within images, exemplified by identifying the precise location of a tiger in a scene. Subsequent weeks will explore advanced segmentation techniques and pose estimation, expanding the scope of computer vision applications. Students are encouraged to review the relevant chapters and apply their knowledge by constructing a ResNet-type model for classifying a set of images with ten distinct classes. The next session is scheduled for the following Monday at nine o'clock.
