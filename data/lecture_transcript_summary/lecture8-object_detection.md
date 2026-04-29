# object detection

The lecture will cover object detection, a subfield of computer vision, focusing on identifying and classifying objects within images or videos. Techniques for object detection include image classification and object tracking. Examples will illustrate these concepts.

## Counting

Object detection in computer vision tasks involves identifying and labeling objects within images, such as people, horses, and ties, and marking their locations with bounding boxes. For tracking, the center of mass of objects is determined in each frame to trace their movement across a video sequence. This technique is applicable to CCTV footage analysis of crowded spaces.

## Tracking

In real-time video analysis, models such as the original video, resonant architectures, and YOLO (YOLO Look Once) are employed. These models face challenges in accuracy, exemplified by potential mislabeling of objects like a frisbee. Object detection involves the identification and precise localization of objects within video frames, a task that becomes increasingly complex at higher frame rates, such as 60 frames per second.

## Object detection is all about the bounding box

The next three weeks focus on deep learning applied to image inputs, specifically on object detection tasks such as image classification, object detection with bounding boxes, and person counting. Object detection involves defining a bounding box, which is the maximum extent of a two-dimensional object, using four numbers to specify its coordinates: the starting location in X and Y, which gives its width and height. Alternatively, the center of the image and its width and height can define a bounding box. In object detection, the goal is to predict these four numbers for each object in an image, which can include multiple objects.

Pixel-level image segmentation, where every pixel is classified into a label, can be used for object detection. However, this method is slow and less efficient for creating training data compared to drawing bounding boxes around objects.

## So how does object detection work in practice?

Two-stage object detection involves a two-step process, while single-stage object detection focuses on a single, more direct approach.

## Two-stage object detection

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

A region proposal algorithm combined with a convolutional neural network (CNN) is used for object detection. Initially, an image is processed to propose regions, which are essentially bounding boxes around potential objects. These regions are then classified by passing them through a CNN. However, this approach can be inefficient due to the large number of regions and the computational cost of processing each through a CNN.

To address these issues, single-stage object detection methods have been developed. These methods predict bounding boxes and labels simultaneously, reducing the number of calls to the CNN and improving efficiency. While this approach may slightly reduce accuracy, it is more suitable for real-time applications, such as in self-driving cars, where speed is crucial.

## Several single-stage object detectors exist

The upcoming lectures will delve into YOLO (You Only Look Once) models, a class of object detection systems. The focus will be on understanding the model's architecture, from training data to image or video classification. The training data for YOLO models typically consists of annotated images or videos where objects of interest are marked with bounding boxes.

## How does training data look like?

The effectiveness of a model in object detection is contingent upon the quality and diversity of its training data. Typically, this data comprises images with various objects, each annotated with bounding boxes and labels. Common objects in these datasets include cows, chairs, cats, cars, and buses. Datasets such as Pascal and COCO are prevalent, with COCO offering a broader range of objects, including cats and bananas, which can be filtered for specific image types. Bounding box coordinates, rather than pixel-level segmentation, are used for object detection, stored in dictionaries for each object. The COCO dataset, for instance, provides images with multiple objects, each

## How does the original YOLO model work?

The focus of the upcoming three weeks is on deep learning applications in image inputs, specifically in computer vision tasks such as image classification, object detection with bounding boxes, and person counting.

In object detection, single-stage methods aim to simultaneously predict bounding boxes and the confidence level of each box containing an object. For each proposed box, the model must predict four coordinates and a confidence score. The model also assigns each image cell to a prediction, typically using a grid system like a 7x7 grid. Each cell predicts one or two boxes, their coordinates, and confidence scores. The output tensor's shape is S by S, where S is the number of cells, and the model predicts 5B values, with B being the number of bounding boxes plus the number of classes. This approach is more complex than single-class image classification but is crucial for understanding object detection mechanisms.

## Some notes on the confidence score

Confidence in object detection models is a measure that combines the probability of an object's presence with the precision of its localization. It is calculated as the product of the object's presence probability and the Intersection over Union (IoU) metric. IoU quantifies the overlap between the predicted bounding box and the ground truth, with a higher IoU indicating a more accurate prediction. A high confidence score suggests both the presence and precise localization of an object within the image.

## Network architecture

The next three weeks focus on deep learning applied to image inputs. Object detection networks, which are essentially convolutional neural networks, typically consist of a series of convolutional layers followed by fully connected layers. These networks are often pre-trained on large datasets like ImageNet and then fine-tuned on specific datasets such as Pascal or Common Objects in Context. Pre-training and fine-tuning are standard practices in object detection models. An alternative approach involves using pre-trained backbones like ResNet or EfficientNet, which can be quickly adapted to new tasks by rescaling input images.

## Choosing a suitable loss

The next lecture will cover the loss functions used in object detection tasks within deep learning. The three components to predict are: classifying objects, placing bounding boxes around them, and determining the model's confidence in the presence of an object.

For object classification, the categorical cross-entropy loss function is employed. For bounding box placement, a common approach is to use the sum of squared differences between the true and predicted coordinates, which are four numbers representing the box's position. This method is analogous to mean squared error and is chosen because large discrepancies result in large loss values.

The confidence score for bounding boxes is calculated using the product of the predicted probability of an…

## Putting it all together

The loss function in deep learning models for image inputs combines three components: coordinate loss, confidence loss, and classification loss. Coordinate loss calculates the error in bounding box coordinates, confidence loss measures the discrepancy between predicted and target confidence levels, and classification loss uses cross-entropy for object classification. These components are summed over all image cells, with the possibility of multiple bounding boxes per cell. Customization of loss components is possible, allowing scaling based on dataset challenges. The authors of a referenced paper emphasized the importance of coordinate loss by scaling it by a factor of 5 to address its complexity.

## Overlapping bounding boxes

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting. In object detection, the model predicts boxes around objects in an image. A specific loss function is used to train the model to perform this task. However, when objects overlap, such as multiple bounding boxes around a single object, the model must discard the boxes with the lowest confidence scores to maintain the most accurate representation. This issue is particularly challenging when objects are densely packed, as the model may struggle to distinguish between them. Adjusting the threshold setting can help manage this complexity.

## Machine learning is learn by doing

The upcoming weeks will delve into deep learning for image inputs, focusing on computer vision tasks such as image classification, object detection with bounding boxes, and person counting. It is recommended to follow the example provided in the book, using common objects from the context dataset to set up a model efficiently. The Cocoa dataset stands out as the premier dataset for object recognition and image segmentation tasks.

## Summary

The next three weeks focus on single-stage object detection in image inputs, which predicts bounding boxes, labels, and confidence scores for objects in real-time applications such as autonomous vehicles and surveillance systems. These detectors are based on convolutional neural networks (CNNs) and often utilize pre-trained models or backbones from datasets like ImageNet for feature extraction.
