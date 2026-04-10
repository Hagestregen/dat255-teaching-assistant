# project kickoff and development tools

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

The lecturer has been working on cloud computing for an extended period, with a focus on Python rather than Java. The group consists of individuals who have not previously met, and the lecturer is preparing for the first project week, potentially followed by a second project week after Easter.

## Project kickoff

The upcoming weeks will focus on selecting and executing a deep learning project. Projects should be feasible within a reasonable timeframe and based on available data. The assignment requires a project description, with the option to use a dataset from the project catalog. Creativity in project design is encouraged, with a submission deadline set for next Friday.

## Assignment rules and grading criteria

The course grade consists of a 50-50 split between an exam and project work. The project aims to demonstrate mastery of the course material. The report should reflect the application of class content and learning. Mandatory items are required, with recommended ones offering grade improvement. Submission of the report in either Norwegian or English is necessary.

## Evaluation criteria

Evaluation criteria will be used to judge the final outcome.

## Requirements for the submission

The upcoming assignments require students to submit their work through WiseFlow. The submission includes a single PDF report and code, which must be hosted on GitHub or a similar platform. The report should contain a link to the code repository, and if applicable, a link to the deployed web application. The PDF submission will be graded on WiseFlow, not on Canvas.

## Formalities

The grading policy is based on a 50-50 system, where the final grade is rounded up. A project grade of C and an exam grade of B result in a final grade of B, unless either is failed. It is crucial to maintain focus throughout the project and prepare adequately for the exam to secure a good outcome. Submission dates were determined through a poll, and the final decision rests with the exam office, with no possibility of extensions. Students must submit doctor's notes directly to the exam office and are permitted to submit multiple drafts for their reports.

## Scope and expectations

The upcoming three weeks will focus on deep learning applications in image processing. The tasks include image classification, object detection with bounding boxes, and counting individuals within images. Students are expected to identify a practical use case for deep learning, design and train neural networks from scratch, and demonstrate their understanding by training a model with randomly initialized weights. They should also compare their models to pre-trained ones, experiment with modifications, and evaluate their models using selected metrics. Additionally, students are encouraged to deploy their models as web applications using Python libraries such as Gradio or Streamlit. Theoretical exploration of deep learning models is encouraged, though practical application through web app development is also recommended for academic improvement.

## Submission format

The report template on Canvas specifies the expected content.

## Project report template

The upcoming three weeks will concentrate on deep learning techniques for image processing. The focus will be on computer vision tasks such as image classification, object detection with bounding boxes, and counting individuals within images.

For the project, the candidate is expected to identify the problem to be solved and select appropriate deep learning models for experimentation. Success criteria will be defined, including the effectiveness of the chosen models and the quality of the implemented solutions.

The candidate should provide a brief overview of the data used, discuss the rationale behind the choice of specific layers or architectures, and include a graphical representation of the network architecture. Code implementation will be evaluated, with an emphasis on the functionality rather than stylistic perfection.

Evaluation will involve presenting numerical results, graphs, and any other relevant findings. If applicable, a web application will be described, with a screenshot and a discussion of its features.

The conclusion should reflect on the project's success or failure, with an explanation of the outcomes and potential areas for improvement.

Questions regarding code sourcing, AI tool usage, pre-trained models, and web app integration will be addressed, with guidelines provided for each.

All submissions must be in PDF format, with code and web app links included in the report.

## Next week: Project week

The course schedule for the upcoming weeks includes a focus on natural language processing, with an emphasis on developing chatbot-type models. Advanced topics such as generative deep learning will be explored, with potential adjustments to the timeline. The project phase will commence in the second week of April, allowing for project finalization.

## Remember to think about your future

Deep learning offers various career paths, including master programs in software development and applied engineering, computing, and engineering. The application deadline for these programs is April 15th.

## The ML project lifecycle

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

The machine learning project life cycle consists of 158 steps, which can be visualized in a graph. This cycle involves data collection and processing, data modeling, and iterative improvement of the machine learning model. For the project, understanding the problem and planning to solve it is essential. Data cleaning is a significant aspect of the process. Feature engineering and model selection, including fine-tuning and ensemble creation, are critical steps. The project may culminate in developing an application to test its utility for end users. Flexibility in focusing on different aspects of the project is encouraged. Testing various machine learning models, such as convolutional neural networks for image classification, is a common practice. Keeping detailed records of experiments and work is crucial for reproducibility.

## Experiment tracking

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.


During the training of deep learning models, it is essential to monitor performance to avoid wasting time on non-functional models or those that overfit. Loss curves are a straightforward tool for diagnosing model issues, indicating whether the model is improving or if training and validation data diverge. Checkpoints are crucial for restoring model weights in case of failure.


Experimentation with deep learning models requires practical tools for performance monitoring. It is beneficial to review the model during training, especially when experimenting with new configurations. Comparing results across multiple models is necessary to identify the most effective ones. Documenting the configuration and settings of successful models helps understand the contributing factors to their success. Determining the reasons behind a model's effectiveness is a complex but essential task.

## Fundamentals of machine learning

The upcoming weeks will delve into deep learning with a focus on image inputs. Key areas of interest include image classification, object detection with bounding boxes, and person counting.

In the realm of machine learning, monitoring loss curves during training is essential. Attention should be given to various tips for effective training. A critical concept to consider for publication-worthy models is ablation studies. This involves systematically removing components from a model to assess their impact on performance. For instance, conducting an ablation study on a model with residual connections might involve sequentially removing layers to determine their significance. This method provides insights into the model's architecture and its contribution to overall performance.

Training performance evaluation is a fundamental step, followed by the development of a model that surpasses previous iterations. This process involves making informed adjustments to the model, guided by the relationship between hyperparameter choices and performance outcomes. The challenge lies in optimizing thousands of hyperparameters efficiently, a topic that will be briefly explored.

The core of experimentation in machine learning revolves around making strategic choices to enhance model performance without exhaustive trial and error.

## Tools for experiment tracking

The lecture will cover experiment tracking tools, with TensorBoard as a standard option. TensorBoard, related to TensorFlow and Keras, facilitates visualization of loss curves, accuracy, and gradients. While it's user-friendly, its development has slowed, limiting advanced features. For more sophisticated tracking, alternative tools are recommended.

## Keras

"To visualize the training process, store callback outputs and access the graphs at localhost:8888. The graphs resemble standard deep learning training curves."

## TensorFlow

The upcoming weeks will cover deep learning tools for experiment tracking and hyperparameter optimization. TensorBoard offers a straightforward installation and use, while Weights & Biases requires a commercial account, though it's free for students. Aim is an open-source alternative that necessitates local execution. These tools are essential for effective deep learning experimentation.

## Hyperparameter optimisation

"Hyperparameter optimization in neural networks is crucial, as it is not directly differentiable. Gradient descent is used for network parameters, which are differentiable. To optimize hyperparameters, one must test various configurations, a process that can be resource-intensive. Efficient experimentation strategies are essential to maximize information from each trial. The Google deep learning tuning playbook provides guidance on this topic."

## Machine Learning

"Optimization of the training process involves testing various hyperparameters, such as learning rates, to enhance model performance."

## Hyperparameter optimisation: Grid scan

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

Hyperparameter optimization in machine learning can be approached through grid search, where a fixed grid of hyperparameter values is tested. This method is straightforward but becomes impractical with an increasing number of hyperparameters due to the curse of dimensionality. The distances between points in the hyperparameter space grow, requiring a denser grid and more experiments to cover the space effectively.

Random search is an alternative that selects hyperparameter values at random. This method can be more efficient than grid search, especially when the number of hyperparameters is large, as it does not require sequential testing and can be parallelized. While it may not guarantee finding the absolute best hyperparameters, it often finds good ones more quickly.

For a more sophisticated approach, one can use the results from previous experiments to inform the selection of hyperparameters for subsequent tests, potentially employing different optimization strategies.

## Hyperparameter optimisation: Intelligent choices

Gradient-free optimization is a popular method for optimization that includes techniques such as evolutionary algorithms and Bayesian optimization. These methods start with an initial experiment and iteratively refine the approach based on the success of previous experiments. They rely on historical data to guide future experiments, which can limit parallel execution due to the sequential nature of the process. Despite this, they are often used due to computational resource constraints.

Several frameworks are available for gradient-free optimization, with some offering higher efficiency despite potential limitations in parallel execution. Search algorithms are commonly recommended for finding optimal model parameters. Gaussian processes, for example, are used in regression to minimize uncertainty by selecting promising locations for testing.

Sequential models can utilize Keras Tuner, which integrates with Keras to suggest hyperparameter values. By training the model repeatedly, Keras Tuner refines its suggestions. Users can experiment with adding layers suggested by Keras Tuner to improve model performance.

Optuna is another tool for optimization that provides numerical suggestions and integrates with Visual Studio Code for visualization. It helps identify the best configurations by generating performance graphs. Analyzing the importance of hyperparameters is essential, and tools like Optuna can indicate which parameters significantly impact performance.

Optuna also allows for recording and visualizing the performance of different experiments, aiding in the management and analysis of results. This can reveal the training duration of models and the relative importance of various

## Hyperparameter Relationships

The lecture covers optimization tools for hyperparameter tuning in machine learning, with a focus on Ray and Keras. Ray allows users to define a search space and execute optimization, while Keras offers various optimization methods suitable for different scenarios. The lecture emphasizes the importance of customizing models as one's technical expertise grows, referencing Chapter 7 for detailed guidance.

## Custom callbacks

Callbacks in deep learning frameworks like Keras provide mechanisms to monitor and influence the training process. They can save models periodically, revert to the best-performing model, halt training upon overfitting, and log various metrics. The model checkpoint callback saves the model at certain intervals, while early stopping halts training when performance plateaus. Learning rate adjustments can be made dynamically to optimize convergence. Keras offers a callbacks API, and users can subclass Keras.callbacks.Callback to implement custom actions at the end of training, such as evaluating test images or integrating custom metrics.

## Custom metrics

"For deep learning tasks, Keras provides built-in metrics like accuracy for classification and regression metrics. Image segmentation tasks have multiple options, and custom metrics can be implemented by defining functions such as `updateState` and `result`. Keras automatically triggers these metrics during training, and scikit-learn offers additional metrics that can be integrated with Keras. ChatGPT can also aid in calculations for custom metrics."

## Custom loss functions

Loss functions are essential for evaluating model performance, comparing predicted outputs to true values. They must be differentiable to enable gradient-based optimization. Mean squared error (MSE) is a common loss function, which can be efficiently implemented using vectorized operations in libraries like TensorFlow, avoiding slow loops or conditional statements. Keras provides similar functions in `keras.ops`, which are optimized for different computational backends.

## Custom layers

The next section will delve into the implementation of custom layers in neural networks, focusing on the integration of weights and learning capabilities. Custom layers are defined by three functions: the initializer, the build function, and the call function. The call function is responsible for the computation, while the build function initializes the weights. For a standard dense layer, the implementation is straightforward. Customization is possible for specialized requirements. The trainable parameter determines whether the weights are optimized during training or remain constant.

## Custom models

The next three weeks will focus on deep learning applied to image inputs. The use of functions and classes in Keras simplifies model architecture customization and construction. Keras's call function streamlines the process of defining complex models by automatically handling the computational graph, differentiation, and training.
