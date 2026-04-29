---
source: "Deep Learning with Python"
---

# Chapter 10: Interpreting what ConvNets learn

A fundamental problem when building a computer vision
application is that of interpretability: Why did your classifier think
a particular image contained a fridge, when all you can see is a truck?
This is especially relevant to use cases where deep learning is used to complement
human expertise, such as medical imaging use cases.
This chapter will get you familiar with a range of different techniques for visualizing
what ConvNets learn and understanding the decisions they make.

It’s often said that deep learning models
are “black boxes”: they learn representations that are
difficult to extract and present in a human-readable form. Although this is
partially true for certain types of deep learning models, it’s definitely not
true for ConvNets. The representations learned by ConvNets are highly
amenable to visualization, in large part because they’re
representations of visual concepts. Since 2013, a wide array of techniques
has been developed for visualizing and interpreting these representations. We
won’t survey all of them, but we’ll cover three of the most accessible and
useful ones:

Visualizing intermediate ConvNet outputs (intermediate activations) —
Useful for understanding how successive ConvNet layers transform their input,
and for getting a first idea of the meaning of individual ConvNet filters

Visualizing ConvNets filters — Useful for understanding precisely what
visual pattern or concept each filter in a ConvNet is receptive to

Visualizing heatmaps of class activation in an image — Useful for
understanding which parts of an image were identified as belonging to a given
class, thus allowing you to localize objects in images

For the first method — activation visualization — you’ll use the small ConvNet that
you trained from scratch on the dogs-versus-cats classification problem in
chapter 8. For the next two methods, you’ll use a pretrained Xception model.

Visualizing intermediate activations

Visualizing intermediate activations
consists of displaying the values returned by various convolution
and pooling layers in a model, given a certain input (the output of a layer
is often called its activation, the output of the activation
function). This gives a view into how an input is decomposed into the
different filters learned by the network. You want to visualize feature maps
with three dimensions: width, height, and depth (channels). Each channel
encodes relatively independent features, so the proper way to visualize these
feature maps is by independently plotting the contents of every channel as a
2D image. Let’s start by loading the model that you saved in section 8.2:

Next, you’ll get an input image — a picture of a cat, not part of the images the
network was trained on.

Let’s display the picture (see figure 10.1).

To extract the feature maps you want to look at, you’ll create a Keras
model that takes batches of images as input and outputs the activations of
all convolution and pooling layers.

When fed an image input, this model returns the values of the layer activations
in the original model, as a list.
This is the first time you’ve encountered a
multi-output model in this book in practice since you learned about them in chapter 7:
until now, the models you’ve seen have had exactly one input and one output.
This one has one input and nine outputs — one output per layer activation.

For instance, this is the activation of the first convolution layer for the cat
image input:

It’s a 178 × 178 feature map with 32 channels. Let’s try plotting the sixth
channel of the activation of the first layer of the original model (see figure
10.2).

This channel appears to encode a diagonal edge detector,
but note that your own channels may vary because
the specific filters learned by convolution layers aren’t deterministic.

Now let’s plot a complete visualization of all the activations in
the network (see figure 10.3). We’ll extract and plot every channel in each
of the layer activations, and we’ll stack the results in one big grid,
with channels stacked side by side.

There are a few things to note here:

The first layer acts as a collection of various edge detectors. At that
stage, the activations retain almost all of the information present in the
initial picture.

As you go higher, the activations become increasingly abstract and less
visually interpretable. They begin to encode higher-level concepts such as
“cat ear” and “cat eye.” Higher representations carry increasingly less
information about the visual contents of the image and increasingly more
information related to the class of the image.

The sparsity of the activations increases with the depth of the layer: in the
first layer, all filters are activated by the input image, but in the
following layers, more and more filters are blank. This means the pattern
encoded by the filter isn’t found in the input image.

We have just observed an important universal characteristic of the
representations learned by deep neural networks: the features extracted by a
layer become increasingly abstract with the depth of the layer. The
activations of higher layers carry less and less information about the
specific input being seen and more and more information about the target (in
this case, the class of the image: cat or dog). A deep neural network
effectively acts as an information distillation pipeline,
with raw data going in (in this case, RGB pictures)
and being repeatedly transformed so that irrelevant information is filtered
out (for example, the specific visual appearance of the image) and useful
information is magnified and refined (for example, the class of the image).

This is analogous to the way humans and animals perceive the world: after
observing a scene for a few seconds, a human can remember which abstract
objects were present in it (bicycle, tree) but can’t remember the specific
appearance of these objects. In fact, if you tried to draw a generic bicycle
from memory, chances are you couldn’t get it even remotely right, even though
you’ve seen thousands of bicycles in your lifetime (see, for example, figure
10.4). Try it right now: this effect is absolutely real. Your brain has learned
to completely abstract its visual input — to transform it into high-level visual
concepts while filtering out irrelevant visual details — making it tremendously
difficult to remember how things around you look.

Visualizing ConvNet filters

Another easy way to inspect the filters
learned by ConvNets is to display the visual pattern that each filter is meant
to respond to. This can be done with gradient ascent in input space,
applying gradient descent to the value of the input image of a
ConvNet so as to maximize the response of a specific filter, starting from a
blank input image. The resulting input image will be one that the chosen
filter is maximally responsive to.

Let’s try this with the filters of the Xception model. The process is simple:
we’ll build a loss function that maximizes the value of a given filter in a
given convolution layer, and then we’ll use stochastic gradient descent to
adjust the values of the input image so as to maximize this activation value.
This will be your second example of a low-level gradient descent loop (the first
one was in chapter 2). We will show it for TensorFlow, PyTorch, and Jax.

First, let’s instantiate the Xception model trained on the ImageNet dataset.
We can once again use the KerasHub library, exactly as we did in chapter 8.

We’re interested in the convolutional layers of the model — the Conv2D
and SeparableConv2D layers. We’ll need to know their names so we can
retrieve their outputs. Let’s print their names, in order of depth.

You’ll notice that the SeparableConv2D layers here are all named something like
block6_sepconv1, block7_sepconv2, etc. — Xception is structured into blocks,
each containing several convolutional layers.

Now let’s create a second model that returns the output of a specific layer
— a “feature extractor” model.
Because our model is a Functional API model,
it is inspectable: you can query the output of one of its layers and reuse
it in a new model. No need to copy the entire Xception code.

To use this model, we can simply call it on some input data, but we should be
careful to apply our model-specific image preprocessing so that our images
are scaled to the same range as the Xception pretraining data.

Let’s use our feature extractor model to define a function that returns a
scalar value quantifying how much a given input image “activates” a given
filter in the layer. This is the loss function that we’ll maximize
during the gradient ascent process:

A non-obvious trick to help the gradient-ascent process go smoothly is
to normalize the gradient tensor by dividing it by its L2 norm (the square
root of the sum of the squares of the values in the tensor). This ensures
that the magnitude of the updates done to the input image is always within the
same range.

Let’s set up the gradient ascent step function. Anything that involves
gradients requires calling backend-level APIs, such as GradientTape in TensorFlow,
.backward() in PyTorch, and jax.grad() in JAX. Let’s line up all the code snippets for
each of the three backends, starting with TensorFlow.

Gradient ascent in TensorFlow

For TensorFlow, we can just open a GradientTape scope and compute the loss
inside of it to retrieve the gradients we need. We’ll use a @tf.function
decorator to speed up computation:

Gradient ascent in PyTorch

In the case of PyTorch, we use loss.backward() and image.grad to obtain
the gradients of the loss with respect to the input image, like this.

No need to reset the gradients since the image tensor is recreated at each iteration.

Gradient ascent in JAX

In the case of JAX, we use jax.grad() to obtain a function that returns the
gradients of the loss with respect to the input image.

The filter visualization loop

Now you have all the pieces. Let’s put them together into a Python function
that takes a filter index as input and returns a
tensor representing the pattern that maximizes the activation of the specified
filter in our target layer.

The resulting image tensor is a floating-point array of shape (200, 200,
3), with values that may not be integers within [0, 255]. Hence, you need to
post-process this tensor to turn it into a displayable image. You do so with
the following straightforward utility function.

Let’s try it (see figure 10.5):

It seems that filter 2 in layer block3_sepconv1 is responsive to a horizontal
lines pattern, somewhat water-like or fur-like.

Now the fun part: you can start visualizing every filter in the layer —
and even every filter in every layer in the model (see figure 10.6).

These filter visualizations tell you a lot about how
ConvNet layers see the world: each layer in a ConvNet learns a collection of
filters such that their inputs can be expressed as a combination of the
filters. This is similar to how the Fourier transform decomposes signals onto
a bank of cosine functions. The filters in these ConvNet filter banks get
increasingly complex and refined as you go higher in the model:

The filters from the first layers in the model encode simple
directional edges and colors (or colored edges, in some cases).

The filters from layers a bit further up the stack, such as block4_sepconv1,
encode simple textures made from combinations of edges and colors.

The filters in higher layers begin to resemble textures found in natural
images: feathers, eyes, leaves, and so on.

Visualizing heatmaps of class activation

Here’s one last visualization technique —
one that is useful for understanding which parts of a
given image led a ConvNet to its final classification decision. This is
helpful for “debugging” the decision process of a ConvNet, particularly in the
case of a classification mistake (a problem domain called model interpretability).
It can also allow you to locate specific objects in an image.

This general category of techniques is called class activation map (CAM)
visualization, and it consists of producing
heatmaps of class activation over input images. A class activation heatmap is
a 2D grid of scores associated with a specific output class, computed for
every location in any input image, indicating how important each location is
with respect to the class under consideration. For instance, given an image
fed into a dogs-versus-cats ConvNet, CAM visualization would allow you to generate
a heatmap for the class “cat,” indicating how cat-like different parts of the
image are, and also a heatmap for the class “dog,” indicating how dog-like
parts of the image are.
The specific implementation we’ll use is the one described in Selvaraju et al.[1]

Grad-CAM consists of taking
the output feature map of a convolution layer, given an input image, and
weighting every channel in that feature map by the gradient of the class with
respect to the channel. Intuitively, one way to understand this trick is that
you’re weighting a spatial map of “how intensely the input image activates
different channels” by “how important each channel is with regard to the
class,” resulting in a spatial map of “how intensely the input image activates
the class.”

Let’s demonstrate this technique using the pretrained Xception model. Consider
the image of two African elephants shown in figure 10.7, possibly a mother and
her calf, strolling in the savanna. We can start by downloading this image and converting it to a NumPy array, as shown in figure 10.7.

So far, we have only used KerasHub to instantiate a pretrained feature extractor
network using the backbone class. For Grad-CAM, we need the entire
Xception model including the classification head — recall that Xception was
trained on the ImageNet dataset with ~1 million labeled images belonging to
1,000 different classes.

KerasHub provides a high-level task API for common end-to-end workflows like
image classification, text classification, image generation, and so on. A task
wraps preprocessing, a feature extraction network, and a task-specific head into
a single class that is easy to use. Let’s try it out:

The top five classes predicted for this image are as follows:

African elephant (with 90% probability)

Tusker (with 5% probability)

Indian elephant (with 2% probability)

Triceratops and Mexican hairless dog with less than 0.1% probability

The network has recognized the image as containing an undetermined quantity of
African elephants. The entry in the prediction vector that was maximally
activated is the one corresponding to the “African elephant” class, at index
386:

To visualize which parts of the image are the most African elephant–like, let’s
set up the Grad-CAM process.

You will note that we didn’t need to preprocess our image before calling the
task model. That’s because the KerasHub ImageClassifier is preprocessing inputs
for us as part of predict(). Let’s preprocess the image ourselves so
we can use the preprocessed inputs directly:

First, we create a model that maps the input image to the activations
of the last convolutional layer.

Second, we create a model that maps the activations of the last convolutional
layer to the final class predictions.

Then, we compute the gradient of the top predicted class for our input image
with respect to the activations of the last convolution layer.
Once again, having to compute gradients means we have to use backend APIs.

Getting the gradient of the top class: TensorFlow version

Let’s start with the TensorFlow version, once again using GradientTape.

Getting the gradient of the top class: PyTorch version

Next, here’s the PyTorch version, using .backward() and .grad.

Getting the gradient of the top class: JAX version

Finally, let’s do JAX. We define a separate loss computation function
that takes the final layer’s output and returns the activation channel corresponding
to the top predicted class. We use this activation value as our loss,
allowing us to compute the gradient.

Displaying the class activation heatmap

Now, we apply pooling and importance weighting to the gradient tensor
to obtain our heatmap of class activation.

For visualization purposes, you’ll also normalize the heatmap between 0 and 1.
The result is shown in figure 10.8.

Finally, let’s generate an image that superimposes the original
image on the heatmap you just obtained (see figure 10.9).

This visualization technique answers two important questions:

Why did the network think this image contained an African elephant?

Where is the African elephant located in the picture?

In particular, it’s interesting to note that the ears of the elephant calf are
strongly activated: this is probably how the network can tell the difference
between African and Indian elephants.

Summary

ConvNets process images by applying a set of learned filters. Filters from earlier layers detect edges and basic textures, while filters from later layers detect increasingly abstract concepts.

You can visualize both the pattern that a filter detects and a filter’s response map across an image.

You can use the Grad-CAM technique to visualize what area(s) in an image were responsible for a classifier’s decision.

Together, these techniques make ConvNets highly interpretable.

Ramprasaath R. Selvaraju, et al., “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,” arxiv (2019), https://arxiv.org/abs/1610.02391.
[↩]

No part of this publication may be reproduced, stored in a retrieval
system, or transmitted, in any form or by means electronic, mechanical,
photocopying, or otherwise, without prior written permission of the
publisher.