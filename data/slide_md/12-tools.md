<!-- source: 12-tools.html -->
<!-- index-title: 12: Tuesday -->

# DAT255 – DAT255: Deep learning engineering
# DAT255: Deep learning engineering

Lecture 12 – Project kickoff, development tools

sma@hvl.no

---

## Project kickoff

**Project topic:**

Up to you! Be creative but reasonable

Two guidelines:

- Relevant data must exist and be readily available (IRL it never is, but this is not our concern right now)

- Pick something you find interesting and motivating

Projects are approved by sending in short description on Canvas,
 deadline **next Friday, 27 Feb**

---

## Assignment rules and grading criteria

… are now on Canvas:

Note: The final submission will be on WiseFlow. Details to follow later

---

## Formalities

- Project counts for 50% of the final grade
- Exam counts for the other 50%
- You need a passing grade on the project in order to take the exam

- Vote for deadline options here (Apr 10 / Apr 17 / Apr 24 / May 1) - Since the project is graded and goes into the strict WiseFlow system, there is **no possibility for an extension**. (Zero, 0, none)

---

## Scope and expectations

**Mandatory** to do:

- Identify a use case for deep learning methods
- Construct and train a deep neural network - Train atleast *one* model from scratch
- Make an objective evaluation of model performance

**Recommended** to do:

- Compare different methods and strategies
- Make use of tools for effective optimisation/monitoring/experimentation
- Investigate quality of the model prediction

***Very* recommended**: Deploy our model as a web app

*In short:* Show what you’ve learnt in the course

---

## Submission format

We want to see two things:

1. **A report** Motivate your choices, describe the implementation, document your experiments, and analyse the outcome. Was it a success? What could have gone better? A template for the report is provided on Canvas.

1. **The code** Everything needed to reproduce your results must be put in a GitHub repo and made accessible. The code must be understandable The exact evaluation criteria are listed on Canvas.

---

## FAQ 💬

- Can I copy code from StackOverflow? Yes, but you must always cite your sources

- Can I use ChatGPT / Cursor / `<some AI tool>` for coding and report writing? Yes, but you must always cite your sources. (More info at hvl.no)

- Can I use pretrained models? Yes, a long as you also build and train *at least one* model from bottom up

- Can I use API calls to OpenAI / DeepSeek / `<some service>` instead of training my own model? No, but you can use it *in addition* to your own model

---

## *Next week:* Project week 🙌

- **No** lectures, work on the project

- Remember to submit project description

- Help / hints / questions: I will be available both on Zoom and in physical form in lecture & lab hours

- Usual lectures again the week after

---

## Remember to think about your future

Many good options for doing more advanced deep learning

---

## The ML project lifecycle

---

## Experiment tracking 🧑‍🔬

**During training: Monitor and diagnose**

- Is my model still improving?
- Are gradients stable?
- Is the model overfitting?

**After training: Compare and select**

- Which of my models performed best?
- …and most importantly: What was the configuration of the best model?
- *Why* did it work? Consider ablation study

**Before next training: Hyperparameter selection**

- Which new models and hyperparameters should I try?

---

## Tools for experiment tracking

*Some options:* AIM

+ Loads of functionality
 + Integrates with practically all frameworks
 + Open-source, self-hosted (although paid plans exist)

Plus others too: MLflow, Sacred, Comet, Neptune

---

## Hyperparameter optimisation

> How many settings can you tweak in your neural network?

Hyperparameters cannot be optimised via gradient descent

Need to select them through experimentation.

Since one experiment (*one training run*) can be very expensive, new settings should be chosen carefully.

Additional tips:

Consult the Google Deep learning tuning playbook

---

## Hyperparameter optimisation: Grid scan

*Option 1:* Grid scan

Divide the range of each hyperparameter into fixed steps, and test all combinations.

+ Simple

+ Parallelisable

– The higher the dimensionality, the bigger the distance between grid points. Need a big number of experiments to have good coverage

---

## Hyperparameter optimisation: Random search

*Option 2:* Random search

Probabilistic approach: Pick hyperparameter values at random

+ Simple

+ Parallelisable

+ Likely to find a better optimum than grid scan in fewer number of tries

– Information about one experiment doesn’t help selecting the next

Should always prefer this over grid search.

---

## Hyperparameter optimisation: Intelligent choices

*Option 3:* Bayesian optimisation, evolutionary algorithms, and other derivative-free optimisation algorithms

Let the result of one experiment influence the choice of the next one

+ Efficient

+ Several good frameworks exist

– Not parallelisable

Always prefer this, except for the most simple cases

---

## Tools for hyperparameter optimisation

*Some options:*

Ray Tune, which interfaces with KerasTuner, Optuna and other optimisation libraries

```
search_space = {
  'units': tune.randint(32, 512)
}

def model_builder(config):
  ...
  units = config['units']
  model.add(keras.layers.Dense(units=units, activation='relu'))
  ...
```

---

## Customising Keras

Keras typically have most of what we need, but occasionally we have to add functionality ourselves

Things we might need to customise:

- Callbacks
- Metrics
- Loss functions
- Layers
- Model behaviour under training vs inference

See chapter 7

---

## Custom callbacks

A ***callback*** is run before/after certain events during training

*Use cases:*

- Save model checkpoint efter each epoch
- Change learning rate
- Write results to log file or monitoring tool

```
class CustomCallback(keras.callbacks.Callback):

    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
      ...

    # and more!
```

---

## Custom metrics

Need some exotic metric you invented yourself, or that is only available in scikit-learn?

Subclass the abstract `keras.metrics.Metrics` class and implement the following methods:

```
class MyMetric(keras.metrics.Metric):

  def __init__(self, ..., **kwargs):
    super().__init__(**kwargs)

    # class variables
    self.result = self.add_variable(shape=(), initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):

    # this is where the result is calculated
    # doesn't return anything
    result = ...
    self.result.add_assign(result)

  def result(self):
    return self.result
```

---

## Custom loss functions

Custom loss functions can either subclassing `keras.losses.Loss`, or they can simply be regular functions, with two inputs (`y_true`, `y_pred`) and one output.

***Important:***

The loss function must be differentiable.

```
def mean_squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)

model.compile(loss=mean_squared_error)
```

*Also important:*

For performance reasons, avoid loops, conditionals, and most python built-ins.

Write vectorisable code, ideally in the backend framework (i.e. TensorFlow) or using `keras.ops`.

---

## Custom layers

For stateful layers, subclass the abstract `keras.layers.Layer` class.

Need three methods: `__init__`, `call`, and `build`.

```
class Linear(keras.layers.Layer):

    def __init__(self, units=32):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units), initializer="random_normal", trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.linalg.matmul(inputs, self.w) + self.b
```

---

## Custom models

Typically we compose complicated models by writing reuseable blocks of layers, as a function

Occasionally, however, we might want to rewrite it as a class.

```
def conv_block(inputs):
  x = layers.Rescaling(1.0 / 255)(inputs)
  x = layers.Conv2D(128, 3, strides=2)(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)

  return x
```

```
class ConvBlock(keras.layers.Layer):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.conv2d = layers.Conv2D(128, 3, strides=2)
    self.batchnorm = layers.BatchNormalization()

  def call(self, inputs):
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = self.conv2d(x)
    x = self.bathnorm(x)
    out = layers.Activation("relu")(x)

    return out
```
