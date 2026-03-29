<!-- source: 16_simple_tabular_data.ipynb -->

# Model comparison on simple tabular data

In this notebook we will try to predict whether or not a patient has diabetes, based on various diagnostic measurements. This is no simple task, but the data we use to train the model contains only numerical data, so preprocessing-wise we can say that the data is relatively simple. In the next notebook we get over to data that requires more work before we get going.

Our task is (as ususal) to find the best possible model, so we will carry out a comparison between a deep neural network and tree-based alternatives.


```python
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```


### Download the data

The data are described at [openml.org](https://www.openml.org/search?type=data&status=active&id=43582&sort=runs). If we download it directly we don't get the usual CSV file, but rather a file including the description and feature names -- so let's do this one-liner that skips all the non-numerical data at the beginning.


```python
!curl -L https://www.openml.org/data/download/22102407/dataset | awk '/^[0-9]/' > diabetes.csv
```


```python
! head -n 3 diabetes.csv
```


```python
feature_names = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]
```


## Read the CSV file

There are many options for reading and processing CSV files, [pandas](https://pandas.pydata.org/) being the most popular. Numpy is also happy to do it, through the `loadtxt()` function, which can be configured to do various types of preprocessing on the fly. Here we have only numerical data, so no need to configure anything.

We should note that simply writing a CSV file reader yourself, using Python's `open()` and `readline()`, is quick and easy and gives you even more fine-grained control of the preprocessing.


```python
data = np.loadtxt(
    'diabetes.csv',
    delimiter=',',
)
```


Split the data in train, validation and test sets.

For a more serious model comparison we should of course consider a crossvalidation study.


```python
X = data[:,:-1]
y = data[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

print('X_train.shape:', X_train.shape)
print('X_val.shape:', X_val.shape)
print('X_test.shape:', X_test.shape)
```


If we like, have a look at the data in form of a pairwise feature scatter plot. This is easiest done with the Pandas and Seaborn libraries, but you can also skip this step or implement the equivalent in Matplotlib.


```python
import seaborn as sns
import pandas as pd

dataframe = pd.DataFrame(np.column_stack((X_train, y_train)), columns=feature_names)
sns.pairplot(dataframe, hue="Outcome")
```


Normalise the data.


```python
scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
```


## Simple dense network

For starters, we construct a simple dense (feed-forward) network, and train it.


```python
nn_model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[-1],)),
    keras.layers.Dense(32),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation="sigmoid")
])
```


```python
nn_model.compile(
    optimizer="Adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```


```python
history = nn_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=64,
    shuffle=True,
    verbose=1
)
```


Plot the loss and accuracy curves:


```python
_, axs = plt.subplots(ncols=2, figsize=(10,5))

axs[0].plot(history.history['loss'])
axs[0].plot(history.history['val_loss'])
axs[0].set_ylabel('Loss')
axs[0].set_xlabel('Epoch')
axs[0].legend(['train', 'val'], loc='upper right')
axs[1].plot(history.history['accuracy'])
axs[1].plot(history.history['val_accuracy'])
axs[1].set_ylabel('Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].legend(['train', 'val'], loc='upper left')
```


Let's train some more models before we do the final evaluation on the test set.


## Compare to a Random Forest model


Now for our first comparison -- a Random Forest model, using the implementation in Scikit-learn.

No hyperparameter tuning so far, we just run it with default parameters.


```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

val_accuracy = accuracy_score(y_val, rf_model.predict(X_val))
print('Validation accuracy:', val_accuracy)
```


## Compare to a gradient-boosted decision tree model

For a more advanced tree-based model, we turn to gradient boosting. A common high-performance implementation is [XGBoost](https://xgboost.readthedocs.io/en/stable/) (eXtreme Gradient Boosting).


```python
!pip install xgboost
```


```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier()
xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)]
)
```


## Evaluate results on the test set


```python
preds_nn = nn_model.predict(X_test)
accuracy_nn = accuracy_score(y_test, binarize(preds_nn, threshold=0.5))

preds_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, preds_rf)

preds_xgb = xgb_model.predict(X_test)
accuracy_xgb = accuracy_score(y_test, preds_xgb)

print('Accuracy neural network:', accuracy_nn)
print('Accuracy random forest:', accuracy_rf)
print('Accuracy XGBoost:', accuracy_xgb)
```


### <span style="color: red; font-weight: bold;">Exercise:<span>

Build a deep neural network that can beat the other tree-based models. 

Any network architectures are allowed!

