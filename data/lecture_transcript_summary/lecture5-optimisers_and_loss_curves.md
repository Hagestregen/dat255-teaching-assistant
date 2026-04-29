# optimisers and loss curves

The next three weeks focus on deep learning applied to image inputs, with an emphasis on computer vision tasks such as image classification, object detection with bounding boxes, and person counting. The course will progress from straightforward model testing to exploring state-of-the-art computer vision techniques. The curriculum will cover optimization methods and loss curves, essential for understanding and improving model training. Students will learn to create synthetic images and design advanced neural network architectures. The course will also address numerical optimization challenges in deep learning, aiming to maximize pattern recognition for decision-making.

## Loss functions

The next three weeks focus on deep learning applied to image inputs, with tasks including image classification, object detection with bounding boxes, and person counting.

In deep learning, the loss function, or error function, is minimized to improve prediction accuracy. The loss function quantifies prediction errors, with a lower value indicating better predictions. For regression, the mean squared error (MSE) is used, which calculates the average squared difference between predicted and true values. The MSE is differentiable and has a minimum at zero, penalizing larger errors more heavily.

For classification, the cross-entropy loss function is preferred. It measures the dissimilarity between the predicted probabilities and the actual class labels. The cross-entropy loss is zero when predictions are perfect and increases as predictions deviate from the true labels. It is suitable for binary classification and can be extended to multi-class classification scenarios.

Both MSE and cross-entropy loss functions are differentiable, allowing for the optimization of neural networks through gradient descent. The choice between MSE and cross-entropy depends on the specific task, with cross-entropy being more common for classification problems.

## Gradient descent

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

Optimization is a critical and time-consuming aspect of training deep learning models. The process involves iteratively adjusting parameters to minimize loss, with the goal of finding the optimal parameter values. Previously, we discussed the importance of decreasing loss and the role of the learning rate in determining the step size for parameter updates.

## Local minima

The next three weeks focus on deep learning applied to image inputs. The challenges include getting stuck in local minima during optimization and distinguishing between poor network design and poor training. Various strategies to overcome these issues will be discussed.

## Keras

"In the upcoming weeks, we will explore various optimizers in deep learning, including Stochastic Gradient Descent (SGD) and Momentum. The learning rate is a critical parameter that requires tuning for optimal performance. Momentum, or Momentum Nest Drop, is another technique that can accelerate training, although its mechanics are not fully explained. Visualization of these concepts will aid in understanding their impact on model training."

## Optimizers

"The choice between using a model with a large number of parameters and finding the optimal parameters for a neural network is significant. The former may offer a comprehensive solution, while the latter focuses on efficiency and effectiveness."

## Comparing gradient descent algorithms

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

Some algorithms, particularly those employing stochastic gradient descent, exhibit varied convergence behaviors. The red algorithm, for instance, oscillates around a saddle point, failing to reach the global minimum that others achieve. In contrast, the green algorithm takes longer to converge but eventually finds the global minimum. The yellow algorithm, while not the fastest, efficiently navigates the loss landscape to find the optimal solution. These differences arise from the algorithms' approaches to navigating high-dimensional loss landscapes, which are common in deep learning models.

## Momentum optimisation

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

In the context of stochastic gradient descent (SGD), momentum is a technique that accelerates the convergence of the algorithm by incorporating a fraction of the previous update vector into the current update. This concept is analogous to a physical system where a ball gains momentum as it rolls downhill, influenced by gravity. The momentum term helps the algorithm to overcome local minima and continue moving towards the global minimum.

The momentum parameter, typically denoted as γ (gamma), controls the influence of the previous update on the current one. A higher value of γ means more momentum, which can help the algorithm to navigate through flat regions and avoid getting stuck in local minima. However, setting γ too high can lead to overshooting the minimum. A common default value for γ is 0.9, but it can be adjusted based on the specific problem and dataset.

The momentum term is added to the gradient descent update rule as follows:

v_t = γv_{t-1} + η∇f(x_t)
x_{t+1} = x_t - v_t

where v_t is the current velocity (update vector), γ is the momentum coefficient, η is the learning rate, ∇f(x_t) is the gradient of the objective function at iteration t, and x_t is the parameter vector at iteration t.

By incorporating momentum, the algorithm can

## Nesterov accelerated gradient

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

Enhancements in deep learning techniques, such as momentum with directional adjustments, improve computational efficiency. This method, known as "accelerated gradients," allows for faster computations by considering future steps in the optimization process. The concept is detailed in the "Enhance On Deep Learning" book, with illustrative figures demonstrating the strategy.

## AdaGrad (Adaptive Gradient)

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

In the context of optimizers in Keras, adaptive gradient methods like adam and adam w adjust the learning rate for each parameter individually. This is beneficial for parameters that decrease at different rates, allowing for more precise updates. However, these methods are not universally effective, particularly for complex problems.

RMSProp, an advanced optimization technique, uses the root mean square of past gradients to adjust the learning rate. This method is less sensitive to the learning rate and can be combined with other techniques for improved performance. RMSProp is generally preferred over simpler

## Adam (and friends)

The upcoming lectures will cover advanced optimizers in deep learning, focusing on their features, hyperparameters, and integration with regularization techniques. These tools are essential for efficient model training and project development.

## Comparison and guidelines

The selection of optimization methods for neural networks depends on the specific problem at hand. Standard gradient descent, while slower, is generally effective for a wide range of tasks. More advanced techniques, such as adaptive moment estimation, offer faster convergence but may not always reach the global optimum. The max trick and Nesterov acceleration are alternatives that can improve performance. Regularization and stochastic gradient descent with additional tricks can address underfitting. Ultimately, the choice of method should be guided by the problem's unique characteristics, and thorough testing is recommended to find the most effective approach.

## Learning rate and loss curves

The next three weeks focus on deep learning applied to image inputs. Computer vision tasks include image classification, object detection with bounding boxes, and person counting.

During training, the model's performance is assessed by computing the error or loss function on a validation dataset. Initially, large errors are expected, but as parameters are fine-tuned, the validation loss stabilizes, indicating an optimal set of weights and biases. However, in practice, training may start well but then overshoot the minimum, leading to overfitting. This suggests that learning rates should be dynamically adjusted to balance speed and accuracy, avoiding both overshooting and slow convergence.

## Learning rate scheduling

The lecture will cover the implementation of learning rate schedules in deep learning models, focusing on strategies such as step decay, linear decay, and more advanced methods. These techniques aim to optimize the learning process by adjusting the learning rate at predetermined epochs or based on performance metrics. The discussion will include the practical application of these methods using Keras tools and the potential need for custom coding to achieve desired outcomes.
