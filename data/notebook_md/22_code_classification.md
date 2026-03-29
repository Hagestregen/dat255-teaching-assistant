<!-- source: 22_code_classification.ipynb -->

# Code classification

In this notebook we will classify posts on StackOverflow, by which programming language they pertain to.

This dataset is an extract from the public [Stack Overflow dataset](https://console.cloud.google.com/marketplace/details/stack-exchange/stack-overflow), and contains the body of 16,000 posts on four languages (Java, Python, CSharp, and Javascript), which are equally divided into train and test.

The keywords "Java", "Python", "CSharp" and "JavaScript" have been replaced in each post by the word "BLANK" in order to increase the difficulty of this dataset in classification examples.


```python
import os
import tensorflow as tf
import keras
```


## Download the data


```python
url = "https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"

dataset_dir = tf.keras.utils.get_file(
    'stack_overflow',
    origin=url,
    extract=True
)

print('Dataset downloaded to', dataset_dir)
```


Load data as TensorFlow datasets


```python
batch_size = 64
seed = 42   # need this for the train-test split

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    os.path.join(dataset_dir, 'train'),
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed
)

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    os.path.join(dataset_dir, 'train'),
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed
)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    os.path.join(dataset_dir, 'test'),
    batch_size=batch_size
)
```


Chack what the classes are:


```python
for i in range(len(raw_train_ds.class_names)):

    print(f"Label {i} corresponds to {raw_train_ds.class_names[i]}")
```


Print the first few entries, aloong with their true class.


```python
for batch in raw_train_ds.take(1):
    for i in range(3):
        text = str(batch[0][i].numpy())
        label = batch[1][i].numpy()

        print('Text:', text)
        print('Class', raw_train_ds.class_names[label])
        print()
```


### <span style="color: red;">Exercise:<span>

Now you do the rest!

Remember that for these data, removing all non-letters will maybe not be a good idea, since these might give hints to what the programming language the example contains.

But, since the data are scaped from the Internet, some characters have been replaced -- for instance, all `<` symbols are stored as `&lt;` and so on.

How you deal with this is up to you.


```python
def custom_standardization(input_data):

    standardised_data = ???

    return standardised_data
```

