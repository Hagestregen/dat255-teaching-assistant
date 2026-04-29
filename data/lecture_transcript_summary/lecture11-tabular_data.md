# tabular data

Week 8 focuses on deep learning applied to image inputs, with topics including image classification, object detection with bounding boxes, and person counting.

For the project work, students are required to submit their project topics for approval. Voting for project submission dates is available through a provided link. Project topics can be submitted directly to Kano's page.

Next week will involve project work with no lectures scheduled. Students are encouraged to utilize lecture times and lab sessions for questions. The course will transition to natural language processing topics.

Development tools and project guidelines, including rules and evaluation forms, will be covered. Students will learn about handling tabular data, such as CSV files and Excel sheets, for deep learning applications.

Tools and project kickoff will be discussed the following day. Students will have access to notebooks for practical exercises on deploying models into web applications using specified frameworks.

## Gradio (and Streamlit) deployment

## Course notebooks

The deep learning-based prediction app, available in Gradio or Streamlit, serves as a test for projects. Tutorial exercises and an optional tabular data exercise are also provided. The instructor's office room is available for consultation, with the room number posted. Further details and clarification can be found in the notebook, as indicated on the calendar.

## Tabular data

Tabular data, characterized by its structured format in rows and columns, is a common data type in machine learning. This week's focus will be on understanding and applying techniques for handling and analyzing tabular data.

## This week

"The upcoming sessions will cover deep learning with Keras, focusing on data processing and embeddings. Tomorrow, we will delve into development tools for tracking deep learning experiments. This will include a review of Chapter seven's custom Keras techniques and guidance on project management. We will also discuss project evaluation criteria."

## (Some) types of data

The upcoming weeks will cover deep learning applications to image inputs, time series analysis, and the handling of tabular data. Text and code, as specialized forms of text, will also be explored for their classification and generation capabilities.

## Tabular data

The upcoming three weeks will delve into deep learning applications for image inputs. The focus will be on computer vision tasks such as image classification, object detection with bounding boxes, and person counting.

In the context of health data, each data point is considered independent, unlike images or time series where proximity implies relatedness. This independence necessitates alternative approaches for data preprocessing, especially for non-numeric data like blood type diagnosis codes and symptoms. While numerical data like age and visit counts can be directly fed into neural networks, categorical and textual data require preprocessing.

Preprocessing steps for non-numeric data may involve encoding techniques to transform textual information into a format suitable for deep learning models. This is essential before applying advanced techniques like language models, which can handle textual data effectively.

The study will explore how to handle independent data points in health datasets, focusing on preprocessing categorical and textual data for deep learning applications.

## Feature normalisation

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.


In the context of deep learning, maintaining consistent scales of input data is crucial to prevent issues like exploding or vanishing gradients. Normalization techniques, such as scaling data to have a mean of zero and a standard deviation of one, are commonly employed. This process, known as standardization, can be automated using tools like scikit-learn's StandardScaler or implemented as a normalization layer in Keras. It's important to train the standard scaler on the training data and apply it to the test and validation data without including any information from the test data.

## Normalization layer

Implementing deep learning models for image inputs can be optimized by integrating normalization layers early in the model architecture, allowing for simultaneous scaling and training.

## Keras

Variance selection in model training can be approached in two ways. The choice often depends on the specific application, but generally, it tends to improve model performance and rarely causes issues. Nonetheless, initial problems may arise.

## Feature transformations

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

Not very normally distributed. For many variables, such as counting something, it cannot be negative. For instance, if you calculate the typical income of people in Norway, you cannot have negative income, right? Similarly, if you count the number of inhabitants in different Norwegian cities, you often get a distribution with many small values and a few large values, capped at zero, resembling an exponential distribution. Scaling it does not make it look like a normal distribution. Linear operations like multiplication and summation do not change its skewed shape; it remains heavy-tailed. To make it look more like a normal distribution, we can apply a log transformation. This transforms the data, making it more symmetric

## Compare the effect of different scalers on data with outliers

The Scikit-Learn documentation provides a detailed comparison of various scalers, including feature processing methods tailored to specific data characteristics. Outliers, which may result from measurement errors or incorrect data entry, necessitate different handling techniques. While not the focus of this discussion, it's important to recognize that outliers can impact the effectiveness of scaling methods. The standard scaler and max-min scaler are two common approaches to feature scaling, with the former adjusting data to a uniform range and the latter scaling based on the data's maximum and minimum values. The presence of outliers can particularly affect the max-min scaler's performance.

## QuantileTransformer (Gaussian output)

In the upcoming weeks, the focus will be on deep learning techniques for image inputs. The course will cover computer vision tasks such as image classification, object detection with bounding boxes, and person counting.


When dealing with numerical data that are not on the same scale, normalization or standardization can be applied to improve network performance and training speed. In the context of medical data, categorical data like blood types (A, B, AB, O) can be encoded numerically using one-hot encoding or label encoding to facilitate model training.

## Dealing with text and categorical data

The next three weeks focus on encoding techniques for categorical data in machine learning. Ordinal encoding assigns numbers to categories based on an assumed order, which may not always be meaningful, especially in medical contexts. One-hot encoding creates separate columns for each category, indicating presence or absence, which avoids imposing an order but can lead to overfitting with small datasets. For diagnosis codes, treating each condition separately and creating new columns for each diagnosis allows the model to learn correlations without manual definition, though domain knowledge can further refine this process.

## Embeddings

The next few weeks will focus on deep learning for image inputs, including image classification, object detection with bounding boxes, and person counting.

Natural language processing (NLP) can be improved by using embedding techniques. Instead of manually adding columns for each category in a dataset, which can lead to overfitting due to sparse data, embeddings offer a more efficient solution. Embeddings create a new abstract vector space that contains the necessary information, learned through training. This approach allows the model to determine the best columns, resulting in a more manageable data representation.

For example, instead of using integers to represent blood types, embeddings can use floating-point values to create vectors that point in specific directions based on the relationships between categories. This method helps in visualizing and understanding the relationships between different categories.

The goal is to allow the model to learn these encodings automatically, avoiding manual encoding. The number of columns, or the dimensionality of the embeddings, depends on the amount of data available. More data allows for higher dimensionality without

## Network architectures for tabular data

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

For tabular data, the goal is to create an effective model. CSV values or Excel sheets are input, with predictions as output. Convolutional layers are not ideal due to the lack of spatial relation between data points. A dense network can be used, but it may have drawbacks due to a large number of parameters and lack of parameter reuse. Feature engineering can help reduce parameters and improve network design.

Alternatives include wide networks with many nodes or deep networks with many layers, both of which have been validated. The transformer architecture, which powers

## The best model for tabular data

"For tabular data, traditional methods such as random forests or gradient boosting often outperform deep learning. These models generally offer better performance and speed for certain cases. Deep learning excels in specific scenarios but may not be universally superior."
