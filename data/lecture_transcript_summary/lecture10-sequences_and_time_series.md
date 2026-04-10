# sequences and time series

The next two weeks will focus on a project-based approach to deep learning, with lectures canceled to allow for hands-on work. The project will culminate in a mandatory assignment submission.

## Project pre-approval

"For the project, write a concise title and description, list two to three group members, and ensure the project scope is appropriate."

## Requirements

"In the upcoming weeks, the focus will be on implementing deep learning techniques for image processing tasks. The essential step is to clearly identify and describe the data source for the project. This data will be utilized in conjunction with Convolutional Neural Networks (CNNs) to develop and test models. The project submission, which includes a detailed description of the chosen data set and its relevance to the experiments, is due in two weeks. The project catalog is available for reference to inspire project ideas."

## Project

The upcoming weeks will concentrate on deep learning with a focus on image inputs. Project work will commence, with the initial proposal submission followed by task planning and guidance. Chapter 18, "Best Practices for Deep Learning," will be reviewed to understand foundational concepts. The subsequent week will introduce tools for experiment structuring, model optimization, performance visualization, and improvement testing. Week nine will be dedicated to project work, utilizing the discussed tools and materials.

## Project submission date

The upcoming three weeks will focus on deep learning applications in image processing. The course will cover computer vision tasks such as image classification, object detection with bounding boxes, and person counting. Students are required to vote for exam dates by next week, after which the exam office will finalize the schedule. No further changes will be made once the dates are set. Students should choose a date that best fits their schedule and vote accordingly. The exam dates will be available in the student portal. Practice exams will be provided for preparation. The course has transitioned from an oral exam format to multiple written exams, with updates made annually. Previous references to a fast AI library have been replaced by Keras. Students should be aware of changes and unfamiliar terms in the course material

## Exams to use for preparation

The lecture will cover the project initiation phase, including the provision of report templates.

## mal-prosjektrapport

"The upcoming weeks will concentrate on deep learning with image inputs. The report template should be concise, and grading will be based on understanding rather than model performance. Challenging projects are acceptable if explained well."

## Sequences

"The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting."

"The upcoming weeks will concentrate on deep learning for image inputs. Tasks include image classification, object detection with bounding boxes, and person counting."

## Recurrent neural networks (RNNs)

The next three weeks focus on deep learning applied to image inputs, with an emphasis on computer vision tasks such as image classification, object detection with bounding boxes, and person counting.

Recurrent neural networks (RNNs) are specialized networks that incorporate loops, allowing them to maintain a form of memory. Unlike traditional feedforward networks, RNNs process data sequentially, feeding the output from one time step as input to the next. This architecture enables the network to remember previous inputs, which is crucial for tasks involving sequential data.

To implement an RNN, one must introduce an additional set of weights that interact with the network's states. This process involves multiplying the state from the previous time step with a set of weights and adding the result to the product of the current input and its corresponding weights. The sum of these two products forms the input to the next layer. This mechanism allows the network to consider

## Intermezzo: Autoregressive models

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

In the context of time series analysis, before the advent of deep learning, autoregressive models were employed to make predictions based on the value from the same data at the previous time step. This week's notebook aims to predict the number of passengers using public transport in Chicago, leveraging the observed weekly pattern of higher usage on weekdays and lower on weekends, known as seasonality.

The autocorrelation function is utilized to identify this weekly pattern, revealing high correlation with the same day of the previous week and low correlation with the same day of the previous week. This pattern repeats every seven days, indicating a consistent cycle.

To model this, terms for each day of the week are added,

## Improved memory cells

The course will explore recurrent neural networks (RNNs) and their ability to handle tasks similar to traditional statistical models. RNNs, despite their simplicity, face the vanishing gradients problem, which hampers their performance on long sequences. This issue arises because RNNs tend to forget information after approximately 20 time steps, making it difficult to learn long-term dependencies. To address this, techniques such as long short-term memory (LSTM) and gated recurrent units (GRU) are employed. These mechanisms are designed to retain information over longer periods, improving the network's ability to learn from extended sequences.

## The long short-term memory (LSTM) cell

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

LSTM networks address the challenge of long-term dependencies in sequence data. Jürgen Schmidt-Huber and colleagues introduced a model with two states, short-term (S) and long-term (L), to solve the Spanish ingredients problem. The model, consisting of a network within a network with four fully connected layers, processes data and decides which information to retain or discard based on its importance. Training this model is complex, but it allows for effective information flow. Despite its intricacy, it was a state-of-the-art

## The gated recurrent unit (GRU)

The next three weeks focus on deep learning applied to image inputs, with tasks including image classification, object detection with bounding boxes, and person counting.

In the context of recurrent neural networks (RNNs), the gated recurrent unit (GRU) simplifies the architecture by reducing the number of hidden states while maintaining the ability to selectively retain or discard information. This is achieved through the use of gates that control the flow of information, allowing the network to keep track of relevant past information without the complexity of multiple hidden states. The GRU's output is used as layers in the network, and while it may not be a focus on exams, it is a valuable tool for time series analysis in projects.

In summary, GRUs offer a simplified yet effective approach to handling time series data by maintaining a memory state that is updated during both training and testing phases. This memory state enables the network to make predictions based on learned patterns, such as recognizing temperature trends.

## Stacking recurrent layers

Deep learning involves creating intermediate abstract representations that encode information more effectively. Recurrent neural networks (RNNs) differ in that each time step produces an output state. In Keras, setting `return_sequences=True` is necessary when stacking RNN layers to maintain the sequence. Typically, the final layer does not require this setting, allowing a dense layer to be added for the complete model. Stacking layers is essential for structuring the computational graph from inputs to outputs, with the potential to enhance performance significantly.

## Training RNNs

The next three weeks focus on improving deep learning workflows and model training efficiency. Techniques such as using saturating activation functions, like those ranging from -1 to 1 or 0 to 1, can stabilize models and prevent overfitting. Layer normalization is recommended to maintain reasonable values between layers, and recurrent dropouts are useful for adjusting activation functions and tuning settings in RNNs. Despite their benefits, RNNs are slower to train due to their lack of parallelizability, which can be mitigated by unrolling the network if sufficient memory is available.

## Bonus trick #1: CNN processing

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

In deep learning, convolutional neural networks (CNNs) are effective pattern detectors, while recurrent neural networks (RNNs) excel in recognizing patterns over time. Combining these two can enhance the model's ability to recognize patterns and remember past events. Pre-processing steps, such as down-sampling and using gated recurrent units, can improve the input to the recurrent layer. Alternatives to recurrence include using CNNs alone or modifying convolutions to look backwards in time, as seen in WaveNet. Testing the effectiveness of CNNs alone for future predictions is also a viable approach.

## Bonus trick #2: bidirectional RNNs

The next three weeks focus on deep learning applied to image inputs, including computer vision tasks such as image classification, object detection with bounding boxes, and person counting.

Time series data can be processed using bi-directional recurrent neural networks (RNNs) in Keras. These networks are designed to handle sequences where both past and future context are important. For tasks like Google Translate, a sequence-to-sequence model with reversed input sequences can be effective. The bi-directional layer in Keras facilitates this by rearranging the sequence, allowing the RNN to consider earlier parts of the sequence for predictions.

In ensemble learning, combining predictions from multiple models can enhance performance. Bi-directional RNNs can be particularly useful for tasks that require understanding the entire sequence, such
