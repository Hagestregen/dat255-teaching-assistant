# image segmentation

The lecture series progresses into week six, focusing on image segmentation, a concept briefly introduced in the previous week. The lecturer, Thalia, will conduct the guest lecture on the following day. The course material for this week includes two chapters on image segmentation and object detection. Students are provided with notebooks to practice these concepts, with the first set of exercises following the textbook and the second set being more hands-on. The lecturer also notes the need to update the lecture title to reflect the current week's topic.

## Computer vision tasks

The next three weeks focus on deep learning applied to image inputs, with an emphasis on computer vision tasks such as image classification, object detection with bounding boxes, and person counting.

In image classification, the task involves selecting one of several possible classes for a given image. This is known as single label multiclass classification. For example, given an image of people biking, the model should predict the class 'biking'.

Multi-label classification allows for multiple labels to be assigned to a single data point. For instance, an image might be labeled with 'bike', 'person', and 'tree', but not 'car', 'house', or 'boat'. This requires a softmax activation function to ensure that the sum of probabilities for all predicted labels equals 100%.

Image segmentation is another task where each pixel is classified into a certain class, such as coloring bikes red and people blue. Object detection involves identifying objects and drawing bounding boxes around them, which can be used for counting people or other objects in an image.

## Image segmentation

Image segmentation techniques include semantic, instance, and panoptic segmentation. Semantic segmentation classifies pixels into categories, instance segmentation distinguishes individual objects, and panoptic segmentation combines both, requiring predictions for each pixel in the image.

## Building a semantic segmentation model

The upcoming weeks will focus on deep learning for image inputs, specifically on tasks such as image classification, object detection with bounding boxes, and person counting.

In the context of deep learning, the initial step involves collecting training data, which includes images and their corresponding ground truth annotations. For the notebook exercises, there are three possible output labels, but the number of categories can vary, and sometimes a binary yes/no response is sufficient. The foreground in an image typically represents the subject of interest, while the background contains all other elements. Predicting the precise contour of the subject is a common goal, although it may not always be feasible after separating the foreground from the background.

To automate the annotation process, various image pools can be utilized, but manual annotation is still a significant part of the workflow. The purple area in the image represents the foreground, and the yellow area indicates the contour or outline of the subject. These areas are essentially images,

## Defining the model

The focus of the upcoming three weeks is on deep learning for image inputs. The course will cover computer vision tasks such as image classification, object detection with bounding boxes, and person counting.

In the context of deep learning, the input is defined as an image with a certain size and color channels. Preprocessing steps include rescaling pixel values from an 8-bit range to a 0 to 1 range to facilitate model training. The encoding phase involves two-dimensional convolutions, which recognize patterns hierarchically. By using strides, the convolutional layers can downsample the image, reducing its dimensions while capturing larger patterns.

The use of strides instead of max pooling allows for larger receptive fields and a more efficient downsampling process. As the image is downsampled through convolutional layers, the model can recognize more complex patterns. The final model uses a Keras framework to encode the image data. The model summary reveals a reduction in image dimensions from 200x200 to 25x25 through successive convolutional layers.

The challenge lies in decoding the low-resolution output back to a high-resolution image that retains the necessary details. This process is crucial for tasks that require

## Transposed convolution

Transposed convolution, also known as deconvolution, reverses the operation of convolution by expanding a single pixel into multiple pixels. This technique is used in image processing to upscale images. It involves a kernel that, when applied to an input pixel, generates an expanded output area. For example, a 2x2 kernel applied to a single pixel can expand it into a 3x3 area. This process is learned during training and allows for the decoding of a compressed image back to its original size.

The encoding and decoding process involves repeating transposed convolution six times with matching stride and kernel size. The model's output consists of three classes: cats, backgrounds, and contours, with softmax applied to ensure the sum of outputs equals one. The model maintains input and output dimensions,

## Improved segmentation models: the U-Net

The upcoming lectures will explore advanced deep learning techniques for image processing, with a focus on U-net architecture and residual connections. U-net is a segmentation model that has proven effective in medical imaging, such as tumor detection in x-rays. It employs a U-shaped structure that compresses input images through encoding steps and then expands them during decoding, with residual connections facilitating information flow and mitigating vanishing gradients. This architecture is adaptable to various applications, including cell tracking.

## Net use cases

The segmentation technique is widely applied in various fields, including medicine, agriculture, and autonomous vehicles. In medicine, it aids in identifying specific areas of interest in medical imaging, such as CT scans of lungs, brain scans for lesions, and MRI scans for diagnostics. In agriculture, it assists farming robots in detecting plant diseases and weeds. Autonomous vehicles utilize segmentation for road marker detection, essential for precise road tracking.

## State-of-the-art image segmentation

The focus is on advanced deep learning techniques for image segmentation, where state-of-the-art methods can handle hundreds of object instances per image. The computational intensity of these methods is notable, yet they enable precise detection of multiple objects. The Segment Anything model by Meta Research exemplifies this capability.

## Segment Anything 2 Demo

The next lecture will explore object detection, with Thalia demonstrating the use of bounding boxes.
