# sequences and time series

The upcoming three weeks will focus on deep learning applied to image inputs, with tasks including image classification, object detection with bounding boxes, and person counting.

In week 7, the course will cover time series and recurrence, emphasizing pattern recognition. Workbooks 12 and 13 have been published, with workbook 14 pending completion. A lecture by Tarja will be held later in the course.

The schedule for the next two weeks is yet to be determined, with week 8 dedicated to project kickoff discussions. Week 9 will finalize project topics, with a mandatory Canvas assignment to select a project topic. Projects will be approved or guided accordingly.

No lectures are scheduled for week 15, allowing students to work on their projects. The project will be graded, and students will have a week off from lectures to focus on their work.

## Timeseries forecasting

Time series analysis involves examining sequences of data points indexed in time order to predict future values or identify patterns.

## Sequences

The next three weeks focus on deep learning applied to image inputs, with tasks including image classification, object detection with bounding boxes, and person counting.

Natural language processing (NLP) is a complex field due to the inherent sequential nature of language. The meaning of sentences can change with the rearrangement of words, as demonstrated by the difference between "I only drink coffee" and "Only I drink coffee." This characteristic of language will be revisited in the context of NLP.

Time series data, which is inherently sequential, will be the subject of today's discussion. Time series data, such as temperature readings taken at regular intervals, must maintain their chronological order to be meaningful. This concept applies to various forms of data, including audio and video, where the sequence of frames or sound waves is crucial for interpretation.

In NLP, understanding the sequence of words is essential for tasks like speech recognition, where the order of words affects the meaning. Similarly, in time series analysis, the order of data points is critical, regardless of the specific timestamps.

Approaches to analyzing time series data include speech recognition, DNA sequence decoding

## Sequence prediction tasks

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting. Time series analysis involves understanding the relationship between neighboring data points, which is crucial for tasks like speech recognition, fraud detection, and medical diagnosis. Forecasting, distinct from classification, involves recognizing patterns to predict future events, such as weather forecasting, energy consumption prediction, stock price prediction, and generative text modeling. Sequence-to-sequence learning, exemplified by language translation, image captioning, and text summarization, requires models to remember context for accurate output generation.

## Sequence classification

The upcoming weeks will explore deep learning for image inputs, focusing on tasks such as image classification, object detection with bounding boxes, and person counting.

In image classification, we analyze images by examining patterns in neighboring pixels, which can be in various directions, providing two dimensions of freedom. The goal is to replicate this analysis along a single axis. Convolutional operations like Conv2D and max pooling, which are effective in two dimensions, can be adapted for one dimension. This adaptation does not limit the input to be univariate; multivariate data, such as temperature and air pressure, can be represented as multiple channels in one dimension.

Understanding image classification techniques enables the application of…

## Convolutions recap (but 1D)

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

In one-dimensional convolutions, the filter is applied across data points, multiplying each position and summing the results. This operation, known as a valid convolution, computes values only where the filter overlaps with the data. Padding can be adjusted to maintain dimensionality ('same') or exclude edge values ('valid'). Max pooling can further reduce dimensionality while preserving significant patterns. Convolutions are essential for pattern recognition, including the detection of opposing patterns, which is crucial for tasks like object detection and person counting.

## Convolution appreciation slide

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

The equation initially presented appears complex but simplifies to a multiplication and summation process, which computers execute efficiently. This pattern detection mechanism, often mistakenly referred to as convolution, is fundamental in signal analysis. By switching the minus sign, the operation remains the same, albeit with a different nomenclature in other fields. The operation's effectiveness is demonstrated through visual examples, showing how it identifies patterns and their opposites, achieving zero correlation for orthogonal signals.

In practical applications, such as image classification and signal processing, convolution proves invaluable. It enables the recovery of digital signals from noise, as demonstrated with Wi-Fi signals, by convolving the noisy signal with a known pattern, resulting in a reconstructed signal that accurately represents the original data. This principle underlies the effectiveness of convolution in pattern recognition and signal processing, highlighting its significance in enhancing digital communications.

## Sequence forecasting: Predicting the future

The next three weeks focus on deep learning applied to image inputs, with an emphasis on computer vision tasks such as image classification, object detection with bounding boxes, and person counting.


In the context of pure convolution networks, the ability to extract patterns is beneficial. These networks assume pattern recognition is location-independent, which is useful for identifying objects like cats in various positions within an image. This concept, known as translation invariance, is also applicable to time series data, such as stock prices, where the goal is to identify trends rather than specific data points. However, predicting future events, like stock prices, requires considering recent data, which is more relevant than historical patterns. The textbook clar

## Seasonality

"Seasonality is a critical factor in temperature data analysis, with higher summer temperatures compared to winter. This concept applies to various scenarios, including daily temperature changes and physical data like bus passenger counts. Accurate predictions require understanding the timestamp, as it helps in interpreting fluctuations and forecasting future values."

## Recurrent neural networks (RNNs)

The next three weeks will focus on deep learning applied to image inputs, with a particular emphasis on convolutional neural networks (CNNs) and recurrent neural networks (RNNs). CNNs are adept at tasks such as image classification, object detection with bounding boxes, and person counting. They operate by applying learned filters to input data, but they do not retain information from new data, which can be a limitation for tasks requiring memory of past inputs.

To overcome this limitation, RNNs introduce a state that represents memory, allowing the model to remember and utilize patterns from previous inputs for future predictions. This concept, known as recurrence, is exemplified by RNNs, which process sequences by maintaining a state that is updated at each time step. The simplest form of RNN involves storing the output from one time step and reintroducing it as input for the next, enabling the network to incorporate past information into its predictions.

Implementing an RNN involves initializing the state to zero and updating it at each time step by combining the current input with the previous state. This process allows for sequence-to-sequence processing, where outputs at each time step can be used for further analysis or as inputs for subsequent time steps. While the basic operation of an RNN is straightforward, involving the multiplication of inputs and states by weights and the addition of biases, more complex RNNs may require additional mechanisms to handle long-term dependencies and improve performance.

In the context of deep learning, both CNNs and RNNs are essential for processing and analyzing sequential and spatial data. CNNs are particularly effective for image-related tasks due to their ability to learn hierarchical representations of visual data. RNNs, on the other hand, are crucial

## Input and output sequences

The lecture discusses sequence prediction in neural networks, emphasizing the difference between tasks that require future predictions and those that benefit from intermediate outputs. For future-oriented tasks, only the final output is considered, while for language prediction, intermediate outputs are valuable. Complex tasks involve encoding and decoding processes, where a state is formed from past information to generate extended sequences, such as summaries or expanded texts.
