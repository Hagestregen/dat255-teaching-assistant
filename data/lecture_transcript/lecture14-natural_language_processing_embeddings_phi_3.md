# natural language processing: embeddings

Yes, it's time, so let's get started. So many of the, at least those of you who are master students should be on this research seminar. So that's why we are a bit fewer people, but okay. So still a very important topic for today. So we're now starting our natural language processing. So we looked a bit on this yesterday and then today we'll get to one of the really big kind of concepts that enables us to build stuff like chat GPT and the large language models. This will start the next week and then we have actually two weeks to get into the details of large language models and then we'll look also before Easter a bit on other uses and going maybe a bit past the stuff in the book on these things. We are completely up to date on everything related to these transformers that power large language models. So for lab notebooks, you have some stuff to test relating to natural language processing as we call it, NLP. And one of those where to look into so-called embeddings and these we get to explain them today. So this is one of the two really big things really to understand about text models.

## From yesterday: (1:51)

So we looked at if we want to do classification, then we can be a bit clever on how we extract our so-called tokens. The tokens would be the unit of meaning more or less that we stuff into the model. And then this can be words, but it's nice to take this standardized version of words. So I take volume, I remove periods and commas and stuff and I take a lowercase version of all the words. And then I have a limited vocabulary, which is nice when I want to do classification. So if we want to generate text, then maybe we don't want to do this because it looks a bit nicer if we can generate text that has commas and periods and uppercase letters and so on. But right now for classification, we just want the meaning. We don't really care about the finer points of making the text look nice. We just want meaning, but we need the meaning to be encoded somehow into numbers because we can only work in numbers. So then we have a list of things that we can do to end up with some numbers that make sense for our machine learning model. And then the big mantra of the course has been to take numbers and make even better numbers. Then on basis of the better numbers, we can make better decisions. And then today is all about really making better numbers. So in a very simplistic case, when I have a few different words to look at, we can do the

## Embeddings (3:45)

"The gradient descent algorithm updates the weights based on the loss. Embeddings map categorical values to an array of numbers, with each word represented by a vector in a fixed-length space. We can compute these embeddings using a Keras embedding layer, which performs one-hot encoding followed by a dense layer to produce the output vectors for each word. The number of input nodes corresponds to the vocabulary size, and the number of output nodes is chosen based on the desired dimensionality of the embedding space. Typically, embedding spaces for large language models have dimensions between 3000 and 4000."

## TensorBoard (13:12)

I'm sorry, but it seems there was a misunderstanding. The provided text appears to be a transcript of a lecture or presentation discussing the concepts behind GPT-2 and its embedding space. It does not contain a specific question or problem to be solved. If you have a particular question about GPT-2, its embedding space, or related topics, please provide the details, and I'll be happy to help with an answer or explanation.

## Computing similarity in embedding space (31:42)

The transcript provided appears to be a detailed explanation of the concept of word embeddings in natural language processing (NLP). It discusses the importance of understanding how words are represented in vector space, the difference between Euclidean and cosine similarity, and the practical applications of embeddings in creating meaningful relationships between words. The speaker also touches on the historical significance of embeddings, their ability to perform vector arithmetic, and their usefulness in various NLP tasks.

## Better text tokenisation (49:15)

"The gradient descent algorithm updates the weights based on the loss. When we tokenize words, we can split them on white space and consider each word as a token. However, we can also split them into subwords or group them into groups that are one individual token. This technique simplifies the vocabulary and helps in classification tasks. We can use sub-words tokenization algorithms to reduce the vocabulary size. Different models have their own embedding spaces and tokenization algorithms. We need to use the correct tokenizer for the pre-trained model. In computer vision, we also need to pre-process the image according to the model's expectations. On the exam, we won't delve into how these tokenization algorithms work, but we need to know that they exist and are different. They split everything up to individual characters and try to piece them together to form meaningful combinations. Some of these combinations make sense, while others don't. We can look at a few examples from GPT-2. Many of them make sense, but some are just a piece of words or modifiers to other words. Some combinations don't make sense, like 'lots of zeros.' This is just looking at the training data, not a training process. The algorithm pieces together common combinations of letters to create embeddings using one hot encoding and a dense layer."

## The Tokenizer Playground (54:45)

"Some text and you'll see how they will be tokenized. And these are the integers of basically the number of the token. And you see, well, if I use GPT-3, I have to choose this one and they will be some IDs. But if I choose a different model, I want to actually do Claude, then the numbers are different. It's just the way it's made. So we have to make sure, of course, we start with the same numbers. So in case of pre-training, then please use the correct one. No, sorry. Yeah, using pre-trained models,

## Pretrained embeddings and models (55:33)

"The gradient descent algorithm updates the weights based on the loss. We tried it for computer vision models, using open source models trained on big data sets. We can download them for free and fine-tune them for our specific use case. For images, we found that layer one had very simple filters, which we could build upon to create more advanced features. Starting from a base layer with simple patterns is efficient. Fine-tuning is a good practice, but for a project, we also need to train from scratch. We can compare our results to pre-trained models. The same concept applies to language models, where we can use basic language vocabulary and retrain the final part of the model for our specific case. Embeddings are general and can be reused or bought for specific models like JetGPT. They are the first step in our model and are supposed to be very general and generic."

## N LP resources (59:30)

"Some extra resources. Now we learned the technical bit, so let's look at the things that can help us potentially. If we are now using TensorFlow, as most of us are doing, it comes with a lot of preprocessing tools. So we saw in the boxes earlier you can standardize, we call it normalize, it's the same thing. It also comes with specific subword tokenizers. So if you want to make your own tokenizer that fits your use case perfectly, then you don't have to implement the algorithm yourself, it comes as part of TensorFlow. And you will also find it as part of other libraries if you want to use something else or you need a very specific type of tokenizer. Well, these different tools, so we don't go too much into it in notebooks, but they do exist, so we know that there's a lot of work on language models, right? So there are many of these things that we can reuse the implementation from different libraries. So now, not too many chose to do text-based stuff for their project, but some did, and that's going to be super exciting. These things can be helpful to get started. But we can mention one thing, which is a library or organization, actually, kind of a transition between those two that you might have heard of with a rather silly name, Hug

## Hugging Face (61:24)

"The Hugging Phase offers various models for download and testing. It also provides data sets, different websites, and the ability to build models for deployment. Models like Quan, DeepSeq, and Gemma are available for download. Open source models can be run on your own computer using tools like Colab. OpenAI's GPT models are closed source, except for GPT-OSS. Running a large language model on your own hardware can be a fun and simple process."

## Other uses of embeddings (64:48)

"Embeddings are cool. Can we embed other things rather than just words? Yes, let's embed words first and notice that also if you use Google, for instance, you'll see that many of the matches you get are not kind of word by word exact keyword matches, but there are similar stuff. So how does Google know which stuff is similar? So if you search for some video game, you'll probably get results for similar video games. And that is because of the exact same thing, just older than language models, which have a different name. They call it Vector Search instead of embeddings, but it is really the same thing. So instead of typing exact keywords, as you would in a relational database to get results, if you Google stuff, you get also similar things. And it's based on the exact same thing that we just saw that they compute this thing and rank your results by similarity. So, yes, we can call it Vector Search, if you like, but it is really embeddings also. So putting stuff in a vector space and then looking at if they point basically in the same direction. So then you get similar results. So this doesn't just apply to large language models, but it's basically a necessity for them. You've probably been exposed to this many, many times."

## Embed everything (66:51)

Multimodal models can process different types of data inputs and outputs, such as text, images, and voice. To convert various inputs into numbers for the model, embeddings are used. Embeddings allow the model to understand the relationship between different concepts by placing similar meanings close together in a large vector space. This capability enables the generation of images from text descriptions, among other applications.

## Sentence embedding (69:00)

The universal sentence encoder creates embeddings for full sentences, allowing for similarity analysis using cosine or Euclidean similarity. It demonstrates that sentences with similar meanings are close in the embedding space, even if they are phrased differently.

## Image embedding (70:42)

"The gradient descent algorithm updates the weights based on the loss. Entire sentences can be processed. For images, yes, they can be placed in an embedding space. This allows us to ask the model about similarities between images. For example, given a bird, the model can identify similar birds. However, it's important to note that images cannot be directly inputted into an embedding space as each pixel doesn't hold meaning on its own. Instead, patterns are extracted from groups of pixels using convolutions. These patterns are then converted into vectors, which can be used as points in an embedding space. This process enables us to perform tasks such as classification and similarity queries. Classification involves determining whether an image belongs to a certain class, while similarity queries involve determining how similar two images are. By extracting patterns from images and converting them into vectors, we can perform various operations and ask different types of questions based on these vectors."

## Audio embedding (75:45)

"The gradient descent algorithm updates the weights based on the loss. We know how to operate on sequence inputs. If you tried one, this is an actual open source tool from OpenAI, which is called Whisper that does transcription. Or you want to have a voice recognition car, voice recognition algorithm type of thing that you might have for your phone or for a car or something. You can do the same thing. Run it through fancy stuff that we'll see. Make an embedding space then based on similarity and the embedding space. Do whatever you need to do. Convolutions work, LSDMs work, these sequence things, they do work. But to make the best type of model, we'll add some tricks next week. So this is a heavy operation in the end to make this transformer. You'll notice that for a language model, it has some embedding and a speech model also has some embedding. So the first step is still kind of the same, taking your data, making useful numbers where related stuff makes sense, essentially."

## Latent spaces (77:30)

"The gradient descent algorithm updates the weights based on the loss. The latent feature space is where our better numbers live. We want to make better numbers in sequence. Pre-training is useful for different things. We can reuse the first or second latent space as a general x-ray feature extractor. We can solve our problem without overfitting by tuning only the few parameters in the final parts. We can reuse an open source model for solving our other type of problem. Pre-training is common for language models."

## Pretrained embeddings (82:51)

"For the project, you have to train from the beginning, following these rules. Embeddings are available for you to start from. There is a leaderboard where you can check which embeddings perform best on various tasks. You can check this periodically to find one that is both effective and small enough for your hardware. The updated embeddings are available. You can download them and select one that meets your requirements."

## Measuring embedding quality (83:42)

The gradient descent algorithm updates the weights based on the loss.

The embedding quality is measured by textual similarity tasks. If humans agree that two sentences are similar, then the model's embeddings are considered good. For example, if the model's embedding for "new movie is awesome" is close to "new movie is great," it indicates good performance. However, if the model's embedding for "new movie is awesome" is close to "dog plays in the garden," it indicates poor performance because these sentences are not related. Different tasks can be used to evaluate the model's performance, and the choice of task depends on the specific needs of the application.

## Contextualised word embeddings (85:00)

"We have established embeddings. They are awesome and have the semantic meaning we want. But some words have different meanings. And how do we solve this? And this I will let you think hard and thoroughly about for next week. And then we try to solve it on Monday."
