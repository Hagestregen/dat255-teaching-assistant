---
source: "Deep Learning with Python"
---

# Chapter 17: Image generation

The most popular and successful application of creative AI today is image generation: learning latent visual spaces and sampling from them to create entirely new pictures, interpolated from real ones — pictures of imaginary people, imaginary places, imaginary cats and dogs, and so on.

## Deep learning for image generation

In this section and the next, we’ll review some high-level concepts pertaining to
image generation, alongside implementation details relative to two of the main
techniques in this domain: *variational autoencoders* (VAEs)
and *diffusion models*.
Do note that the techniques we present here aren’t specific to images — you could
develop latent spaces of sound or music using similar models — but
in practice, the most interesting results so far have been obtained with pictures,
and that’s what we focus on here.

### Sampling from latent spaces of images

The key idea of image generation is to develop a
low-dimensional *latent space* of representations (which, like everything else
in deep learning, is a vector space)
where any point can be mapped to a “valid” image: an image that looks like the real thing.
The module capable of realizing this mapping, taking as input a latent point and
outputting an image (a grid of pixels), is usually called a *generator*,
or sometimes a *decoder*. Once such a latent space has
been learned, you can sample points from it, and,
by mapping them back to image space, generate images that have never
been seen before (see figure 17.1) — the in-betweens of the training images.

Further, *text-conditioning* makes it possible to map a space of prompts in natural language
to the latent space (see figure 17.2), making it possible to do
*language-guided image generation* — generating pictures that correspond to a text description.
This category of models is called *text-to-image* models.

Interpolating between many training images in the latent space enables such models to generate infinite combinations of visual concepts, including many that no one had explicitly come up with before. A horse riding a bike on the moon? You got it. This makes image generation a powerful brush for creative-minded people to play with.

Of course, there are still challenges to overcome. Like with all deep learning models, the latent space doesn’t encode a consistent model of the physical world, so you might occasionally see hands with extra fingers, incoherent lighting, or garbled objects. The coherence of generated images is still an area of active research. In the case of figure 17.2, despite having seen tens of thousands of images of people riding bikes, the model doesn’t understand in a human sense what it means to ride a bike — concepts like pedaling, steering, or maintaining upright balance. That’s why your bike-riding horse is unlikely to get depicted pedaling with its hind legs in a believable manner, the way a human artist would draw it.

There’s a range of different strategies for learning such latent spaces of image representations, each with its own characteristics. The most common types of image generation models are

- Diffusion models
- Variational autoencoders (VAEs)
- Generative adversarial networks (GANs)

While previous editions of this book covered GANs, they have gradually fallen out of fashion in recent years and have been all but replaced by diffusion models. In this edition, we’ll cover both VAEs and diffusion models and we will skip GANs. In the models we’ll build ourselves, we’ll focus on unconditioned image generation — sampling images from a latent space without text conditioning. However, you will also learn how to use a pretrained text-to-image model and how to explore its latent space.

### Variational autoencoders

VAEs, simultaneously discovered by Kingma and Welling in
December 2013[1]
and Rezende, Mohamed, and Wierstra in
January 2014,[2]
are a kind of generative model that’s
especially appropriate for the task of image editing via concept vectors.
They’re a kind of *autoencoder* — a type of network that aims to encode an
input to a low dimensional latent space and then decode it back — that mixes
ideas from deep learning with Bayesian inference.

VAEs have been around for over a decade, but they remain relevant to this day and continue to be used in recent research. While VAEs will never be the first choice for generating high-fidelity images — where diffusion models excel — they remain an important tool in the deep learning toolbox, particularly when interpretability, control over the latent space, and data reconstruction capabilities are crucial. It’s also your first contact with the concept of the autoencoder, which is useful to know about. VAEs beautifully illustrate the core idea behind this class of models.

A classical image autoencoder takes an image, maps it to a latent vector space
via an encoder module, and then decodes it back to an output with the same
dimensions as the original image, via a decoder module (see figure 17.3). It’s
then trained by using as target data the *same images* as the input images,
meaning the autoencoder learns to reconstruct the original inputs. By imposing
various constraints on the code (the output of the encoder), you can get the
autoencoder to learn more or less interesting latent representations of the
data. Most commonly, you’ll constrain the code to be low-dimensional and
sparse (mostly zeros), in which case the encoder acts as a way to compress the
input data into fewer bits of information.

In practice, such classical autoencoders don’t lead to particularly useful or nicely structured latent spaces. They’re not much good at compression either. For these reasons, they have largely fallen out of fashion. VAEs, however, augment autoencoders with a little bit of statistical magic that forces them to learn continuous, highly structured latent spaces. They have turned out to be a powerful tool for image generation.

A VAE, instead of compressing its input image into a fixed code in the latent space, turns the image into the parameters of a statistical distribution: a mean and a variance. Essentially, this means we’re assuming the input image has been generated by a statistical process, and that the randomness of this process should be taken into account during encoding and decoding. The VAE then uses the mean and variance parameters to randomly sample one element of the distribution, and decodes that element back to the original input (see figure 17.4). The stochasticity of this process improves robustness and forces the latent space to encode meaningful representations everywhere: every point sampled in the latent space is decoded to a valid output.

In technical terms, here’s how a VAE works:

-
An encoder module turns the input sample

`input_img`

into two parameters in a latent space of representations,`z_mean`

and`z_log_variance`

. -
You randomly sample a point

`z`

from the latent normal distribution that’s assumed to generate the input image, via`z = z_mean + exp(z_log_variance) * epsilon`

, where`epsilon`

is a random tensor of small values. -
A decoder module maps this point in the latent space back to the original input image.


Because `epsilon`

is random, the process ensures that every point that’s close
to the latent location where you encoded `input_img`

(`z-mean`

) can be decoded
to something similar to `input_img`

, thus forcing the latent space to be
continuously meaningful. Any two close points in the latent space will decode
to highly similar images. Continuity, combined with the low dimensionality of
the latent space, forces every direction in the latent space to encode a
meaningful axis of variation of the data, making the latent space very
structured and thus highly suitable to manipulation via concept vectors.

The parameters of a VAE are trained via two loss functions:
a *reconstruction loss* that forces the decoded samples to match the initial inputs,
and a *regularization loss* that helps learn
well-rounded latent distributions and reduces overfitting to the training data.
Schematically, the process looks like this:

You can then train the model using the reconstruction loss and the regularization loss. For the regularization loss, we typically use an expression (the Kullback–Leibler divergence) meant to nudge the distribution of the encoder output toward a well-rounded normal distribution centered around 0. This provides the encoder with a sensible assumption about the structure of the latent space it’s modeling.

Now let’s see what implementing a VAE looks like in practice!

### Implementing a VAE with Keras

We’re going to be implementing a VAE that can generate MNIST digits. It’s going to have three parts:

- An encoder network that turns a real image into a mean and a variance in the latent space
- A sampling layer that takes such a mean and variance and uses them to sample a random point from the latent space
- A decoder network that turns points from the latent space back into images

The following listing shows the encoder network you’ll use, mapping images to
the parameters of a probability distribution over the latent space. It’s a
simple ConvNet that maps the input image `x`

to two vectors, `z_mean`

and
`z_log_var`

. One important detail is that we use strides for downsampling
feature maps, instead of max pooling. The last time we did this was in the
image segmentation example of chapter 11. Recall that, in general, strides
are preferable to max pooling for any model that cares about *information location* —
that is, *where* stuff is in the image — and this one does,
since it will have to produce an image encoding
that can be used to reconstruct a valid image.

Its summary looks like this:

Next is the code for using `z_mean`

and `z_log_var`

, the parameters of the
statistical distribution assumed to have produced `input_img`

, to generate a
latent space point `z`

.

The following listing shows the decoder implementation. We reshape the vector
`z`

to the dimensions of an image and then use a few convolution layers to
obtain a final image output that has the same dimensions as the original
`input_img`

.

Its summary looks like this:

Now, let’s create the VAE model itself. This is your first example of a model
that isn’t doing supervised learning (an autencoder is an example of *self-supervised*
learning because it uses its inputs as targets). Whenever you depart
from classic supervised learning, it’s common to subclass the `Model`

class
and implement a custom `train_step()`

to specify the new training logic,
a workflow you’ve learned about in chapter 7. We could easily do that here,
but a downside of this technique is that the `train_step()`

contents
must be backend specific — you’d use `GradientTape`

with TensorFlow, you’d use `loss.backward()`

with PyTorch, and so on. A simpler way to customize your training logic is to just
implement the `compute_loss()`

method instead and keep the default `train_step()`

.
`compute_loss()`

is the key bit of differentiable logic called by the built-in `train_step()`

.
Since it doesn’t involve direct manipulation of gradients, it’s easy to keep it backend agnostic.

Its signature is as follows:

`compute_loss(x, y, y_pred, sample_weight=None, training=True)`


where `x`

is the model’s input; `y`

is the model’s target (in our case, it is `None`

since the dataset
we use only has inputs, no targets); and `y_pred`

is the output of `call()`

— the model’s predictions. In any supervised
training workflow, you’d compute the loss based on `y`

and `y_pred`

. In our case,
since `y`

is `None`

and `y_pred`

contains the latent parameters, we’ll compute
the loss using `x`

(the original input) and the `reconstruction`

derived from `y_pred`

.

The method must return a scalar, the loss value to be minimized.
You can also use `compute_loss()`

to update the state of your metrics,
which is something we’ll want to do in our case.

Now, let’s write our VAE with a custom `compute_loss()`

method. It works with all
backends with no code changes!

Finally, you’re ready to instantiate and train the model on MNIST digits.
Because `compute_loss()`

already takes care of the loss, you don’t specify an external loss at
compile time (`loss=None`

), which, in turn, means you won’t pass target data
during training (as you can see, you only pass `x_train`

to the model in `fit`

).

Once the model is trained, you can use the `decoder`

network to turn arbitrary latent space vectors into images.

The grid of sampled digits (see figure 17.5) shows a completely continuous distribution of the different digit classes, with one digit morphing into another as you follow a path through latent space. Specific directions in this space have a meaning: for example, there’s a direction for “four-ness,” “one-ness,” and so on.

In the next section, we’ll cover in detail another major tool for generating images: diffusion models, the architecture behind nearly all commercial image generation services today.

## Diffusion models

A long-standing application of autoencoders has been *denoising*: feeding into a model
an input that features a small amount of noise — for instance, a low-quality JPEG image —
and getting back a cleaned-up version of the same input. This is the one task that autoencoders excel at.
In the late 2010s, this idea gave rise to very successful *image super-resolution* models,
capable of taking in low-resolution, potentially noisy images and outputting high-quality, high-resolution versions
of them (see figure 17.6). Such models have been shipped as part of every major smartphone camera app for the past few years.

Of course, these models aren’t magically recovering lost details hidden in the input,
like in the “enhance” scene from *Blade Runner* (1982).
Rather, they’re making educated guesses about what the image should look like
— they’re *hallucinating* a cleaned-up, higher-resolution version of what you give them.
This can potentially lead to funny mishaps.
For instance, with some AI-enhanced cameras, you can take a picture of something that looks
vaguely moon-like (such as a printout of a severely blurred moon image),
and you will get in your camera roll a crisp picture of the moon’s craters.
A lot of detail that simply wasn’t present in the printout gets straight-up hallucinated by the camera,
because the super-resolution model it uses is overfitted to moon photography images.
So, unlike Rick Deckard, definitely don’t use this technique for forensics!

Early successes in image denoising led researchers to an arresting idea:
since you can use an autoencoder to remove a small amount
of noise from an image, surely it would be possible to repeat the process
multiple times in a loop to remove a large amount of noise.
Ultimately, could you denoise an image made out of *pure noise*?

As it turns out, yes, you can. By doing this, you can effectively hallucinate
brand new images out of nothing, like in figure 17.7.
This is the key insight behind diffusion models, which should more accurately
be called *reverse diffusion* models, since “diffusion” refers to the process
of gradually adding noise to an image until it disperses into nothing.

A diffusion model is essentially a denoising autoencoder in a loop, capable of turning pure noise into sharp, realistic imagery. You may know this poetic quote from Michelangelo, “Every block of stone has a statue inside it and it is the task of the sculptor to discover it” — well, every square of white noise has an image inside it, and it is the task of the diffusion model to discover it.

Now, let’s build one with Keras.

### The Oxford Flowers dataset

The dataset we’re going to use is the Oxford Flowers dataset (https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), a collection of 8,189 images of flowers that belong to 102 different species.

Let’s get the dataset archive and extract it:

`fpath`

is now the local path to the extracted directory.
The images are contained in the `jpg`

subdirectory there.
Let’s turn them into an iterable dataset using `image_dataset_from_directory()`

.

We need to resize our images to a fixed size,
but we don’t want to distort their aspect ratio since
this would negatively affect the quality of our generated images,
so we use the `crop_to_aspect_ratio`

option to extract
maximally large undistorted crops of the right size (128 × 128):

Here’s an example image (figure 17.8):

### A U-Net denoising autoencoder

The same denoising model gets reused across each iteration of the diffusion denoising process,
erasing a little bit of noise each time.
To make the job of the model easier,
we tell it how much noise it is supposed to extract for a given input image — that’s
the `noise_rates`

input. Rather than outputting a denoised image,
we make our model output a predicted noise mask, which
we can subtract from the input to denoise it.

For our denoising model, we’re going to use a U-Net — a kind of ConvNet originally developed for image segmentation. It looks like figure 17.9.

This architecture features three stages:

- A
*downsampling stage*, made of several blocks of convolution layers, where the inputs get downsampled from their original 128 × 128 size down to a much smaller size (in our case, 16 × 16). - A
*middle stage*, where the feature map has a constant size. - An
*upsampling stage*, where the feature map get upsampled back to 128 × 128.

There is a 1:1 mapping between the blocks of the downsampling and upsampling stages: each upsampling block is the inverse of a downsampling block. Importantly, the model features concatenative residual connections going from each downsampling block to the corresponding upsampling block. These connections help avoid loss of image detail information across the successive downsampling and upsampling operations.

Let’s assemble the model using the Functional API:

You would instantiate the model with something like
`get_model(image_size=128, widths=[32, 64, 96, 128], block_depth=2)`

.
The `widths`

argument is a list containing the `Conv2D`

layer sizes for each successive
downsampling or upsampling stage. We typically want the layers to get bigger
as we downsample the inputs (going from 32 to 128 units here) and then get smaller as as upsample
(from 128 back to 32 here).

### The concepts of diffusion time and diffusion schedule

The diffusion process is a series of steps in which we apply our denoising autoencoder to erase a small amount of noise from an image, starting with a pure-noise image, and ending with a pure-signal image. The index of the current step in the loop is called the *diffusion time* (see figure 17.7). In our case, we’ll use a continuous value between 1 and 0
for this index — a value of 1 indicates the start of the process, where the amount of noise is maximal and the amount of signal is minimal,
and a value of 0 indicates the end of the process, where the image is almost all signal and no noise.

The relationship between the current diffusion time and the amount of noise and signal present in the image is called the *diffusion schedule*. In our experiment, we’re going to use a cosine schedule to smoothly transition from a high signal rate (low noise) at the beginning to a low signal rate (high noise) at the end of the diffusion process.

This `diffusion_schedule()`

function takes as input a `diffusion_times`

tensor,
which represents the progression of the diffusion process
and returns the corresponding `noise_rates`

and `signal_rates`

tensors.
These rates will be used to guide the denoising process.
The logic behind using a cosine schedule is to maintain the relationship `noise_rates ** 2 + signal_rates ** 2 == 1`

(see figure 17.10).

Let’s plot how this function maps diffusion times (between 0 and 1) to specific noise rates and signal rates (see figure 17.11):

### The training process

Let’s create a `DiffusionModel`

class to implement the training procedure.
It’s going to have our denoising autoencoder as one of its attributes.
We’re also going to need a couple more things:

*A loss function*— We’ll use mean absolute error as our loss, that is to say`mean(abs(real_noise_mask - predicted_noise_mask))`

.*An image normalization layer*— The noise we’ll add to the images will have unit variance and zero mean, so we’d like our images to be normalized as such too, for the value range of the noise to match the value range of the images.

Let’s start by writing the model constructor:

The first method we’re going to need is the denoising method. It simply calls the denoising model to retrieve a predicted noise mask, and it uses it to reconstruct a denoised image:

Next comes the training logic. This is the most important part!
Like in the VAE example, we’re going to implement a custom `compute_loss()`

method to keep our model backend agnostic. Of course, if you are set on using one
specific backend, you could also write a custom `train_step()`

with the exact same logic in it,
plus the backend-specific logic for gradient computation and weight updates.

Since `compute_loss()`

receives as input the output of `call()`

, we’re going
to put the denoising forward pass in `call()`

. Our `call()`

takes a batch of clean input images
and applies the following steps:

- Normalizes the images
- Samples random diffusion times (the denoising model needs to be trained on the full spectrum of diffusion times)
- Computes corresponding noise rates and signal rates (using the diffusion schedule)
- Adds random noise to the clean images (based on the computed noise rates and signal rates)
- Denoises the images

It returns

- The predicted denoised images
- The predicted noise masks
- The actual noise masks it applied

These last two quantities are then used in `compute_loss()`

to compute the loss of the model
on the noise mask prediction task:

### The generation process

Finally, let’s implement the image generation process.
We start from pure random noise, and we repeatedly apply the `denoise()`

method
until we get high-signal, low-noise images.

### Visualizing results with a custom callback

We don’t have a proper metric to judge the quality of our generated images, so you’re going to want
to visualize the generated images yourself over the course of training to judge if your model is getting somewhere.
An easy way to do this is with a custom callback. The following callback uses the `generate()`

method at the
end of each epoch to display a 3 × 6 grid of generated images:

### It’s go time!

It’s finally time to train our diffusion model on the Oxford Flowers dataset. Let’s instantiate the model:

We’re going to use `AdamW`

as our optimizer, with a few neat options enabled to help stabilize training
and improve the quality of the generated images:

*Learning rate decay*— We gradually reduce the learning rate during training, via an`InverseTimeDecay`

schedule.*Exponential moving average of model weights*— Also known as Polyak averaging. This technique maintains a running average of the model’s weights during training. Every 100 batches, we overwrite the model’s weights with this averaged set of weights. This helps stabilize the model’s representations in scenarios where the loss landscape is noisy.

The code is

Let’s fit the model. We’ll use our `VisualizationCallback`

callback to plot examples of generated images
after each epoch, and we’ll save the model’s weights with the `ModelCheckpoint`

callback:

If you’re running on Colab, you might run into the error, “Buffered data was truncated after reaching the output size limit.”
This happens because the logs of `fit()`

include images, which take up a lot of space, whereas the allowed output for a single notebook cell
is limited. To get around the problem, you can simply chain five `model.fit(..., epochs=20)`

calls, in five successive cells.
This is equivalent to a single `fit(..., epochs=100)`

call.

After 100 epochs (which takes about 90 minutes on a T4, the free Colab GPU), we get pretty generative flowers like these (see figure 17.12).

You can keep training for even longer and get increasingly realistic results.

So that’s how image generation with diffusion works! Now, the next step to unlock their potential is
to add *text conditioning*, which would result in a text-to-image model, capable
of producing images that match a given text caption.

## Text-to-image models

We can use the same basic diffusion process to create a model that maps text
input to image output. To do this we need to take a pretrained text encoder
(think a transformer encoder like RoBERTa from chapter 15) that can map text to
vectors in a continuous embedding space. Then we can train a diffusion model on
`(prompt, image)`

pairs, where each prompt is a short, textual description of
the input image.

We can handle the image input in the same way as we did previously, mapping noisy
input to a denoised output that progressively approaches our input image.
Critically, we can extend this setup by also passing the embedded text prompt to
the denoising model. So rather than our denoising model simply taking in a
`noisy_images`

input, our model will take two inputs: `noisy_images`

and
`text_embeddings`

. This gives a leg up on the flower denoiser we trained
previously. Instead of learning to remove noise from an image without any additional
information, the model gets to use a textual representation of the final image
to help guide the denoising process.

After training is when things get a bit more fun. Because we have trained a
model that can map pure noise to images *conditioned on* a vector
representation of some text, we can now pass in pure noise and a never-before-seen
prompt and denoise it into an image for our prompt.

Let’s try this out. We won’t actually train one of these models from scratch in this book — you have all the ingredients you need, but it’s quite expensive and time consuming to train a text-to-image diffusion model that works well. Instead, we will play with a popular pretrained model in KerasHub called Stable Diffusion (figure 17.13). Stable Diffusion is made by a company named Stability AI that specializes in making open models for image and video generation. We can use the third version of their image generation model in KerasHub with just a couple of lines of code:

Like the `CausalLM`

task we covered last chapter, the `TextToImage`

task is a
high-level class for performing image generation conditioned on text input. It
wraps tokenization and the diffusion process into a high-level generate call.

The Stable Diffusion model actually adds a second “negative prompt” to its
model, which can be used to steer the diffusion process away from certain text
inputs. There’s nothing magic here. To add a negative prompt, you could simply
train a model on triplets: `(image, positive_prompt, negative_prompt)`

, where
the positive prompt is a description of the image, and the negative prompt is
a series of words that do not describe the image. By feeding the positive and
negative text embedding to the denoiser, the denoiser will learn to steer the
noise toward images that match the positive prompt and away from images that
match the negative prompt (figure 17.14). Let’s try removing the color blue from our input:

Like the `generate()`

method for text models we used in the last chapter, we have a few
additional parameters we can pass to control the generation process. Let’s try
passing a variable number of diffusion steps to our model to see the denoising
process in action (figure 17.15):

### Exploring the latent space of a text-to-image model

There is probably no better way to see the interpolative nature of deep neural networks than text diffusion models. The text encoder used by our model will learn a smooth, low-dimensional manifold to represent our input prompts. It’s continuous, meaning we have learned a space where we can walk from the text representation of one prompt to another, and each intermediate point will have semantic meaning. We can couple that with our diffusion process to morph between two images by simply describing each end state with a text prompt.

Before we can do this, we will need to break up our high-level `generate()`

function into its constituent parts. Let’s try that out.

Our generation process has three distinct steps:

- First, we take our prompts, tokenize them, and embed them with our text encoder.
- Second, we take our text embeddings and pure noise and progressively “denoise” the noise into an image. This is the same as the flower model we just built.
- Lastly, we map our model outputs, which are from
`[-1, 1]`

back to`[0, 255]`

so we can render the image.

One thing to note here is that our text embeddings actually contain four separate tensors:

Rather than only passing the final, embedded text vector to the denoising model, the Stable Diffusion authors chose to pass both the final output vector and the last representation of the entire token sequence learned by the text encoder. This effectively gives our denoising model more information to work with. The authors do this for both the positive and negative prompts, so we have a total of four tensors here:

- The positive prompt’s encoder sequence
- The negative prompt’s encoder sequence
- The positive prompt’s encoder vector
- The negative prompt’s encoder vector

With our `generate()`

function decomposed, we can now try walking the latent
space between two text prompts. To do so, let’s build a function to interpolate
between the text embeddings outputted by the model.

You’ll notice we use a special interpolation function called `slerp`

to walk
between our text embeddings. This is short for *spherical linear interpolation*
— it’s a function that has been used in computer graphics for decades to
interpolate points on a sphere.

Don’t worry too much about the math; it’s not important for our example, but it is important to understand the motivation. If we imagine our text manifold as a sphere and our two prompts as random points on that sphere, directly linearly interpolating between these two points would land us inside the sphere. We would no longer be on its surface. We would like to stay on the surface of the smooth manifold learned by our text embedding — that’s where embedding points have meaning for our denoising model. See figure 17.16.

Of course, the manifold learned by our text embedding model is not actually
spherical. But it’s a smooth surface of numbers all with the same rough
magnitude — it is *sphere-like*, and interpolating as if we were on a sphere is
a better approximation than interpolating as if we were on a line.

With our interpolation defined, let’s try walking between the text embeddings for two prompts and generating an image at each interpolated output. We will run our slerp function from 0.5 to 0.6 (out of 0 to 1) to zoom in on the middle of the interpolation right when the “morph” becomes visually obvious (figure 17.17):

This might feel like magic the first time you try it, but there’s nothing magic about it — interpolation is fundamental to the way deep neural networks learn. This will be the last substantive model we work with in the book, and it’s a great visual metaphor to end with. Deep neural networks are interpolation machines; they map complex, real-world probability distributions to low-dimensional manifolds. We can exploit this fact even for input as complex as human language and output as complex as natural images.

## Summary

- Image generation with deep learning is done by learning latent spaces that capture statistical information about a dataset of images. By sampling and decoding points from the latent space, you can generate never-before-seen images. There are three major tools to do this: VAEs, diffusion models, and GANs.

- VAEs result in highly structured, continuous latent representations. For this reason, they work well for doing all sorts of image editing in latent space: face swapping, turning a frowning face into a smiling face, and so on. They also work nicely for doing latent space–based animations, such as animating a walk along a cross section of the latent space, showing a starting image slowly morphing into different images in a continuous way.

- Diffusion models result in very realistic outputs and are the dominant method of image generation today. They work by repeatedly denoising an image, starting from pure noise. They can easily be conditioned on text captions to create text-to-image models.

- Stable Diffusion 3 is a state-of-the-art pretrained text-to-image model that you can use to create highly realistic images of your own.

- The visual latent space learned by such text-to-image diffusion models is fundamentally interpolative. You can see this by interpolating between the text embeddings used as inputs to the diffusion process and achieving a smooth interpolation between images as output.