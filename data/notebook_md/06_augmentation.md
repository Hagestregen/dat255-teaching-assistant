<!-- source: 06_augmentation.ipynb -->

# Augmentation

In this notebook we take a closer look at _augmentation_, and test its effect by training a convolutional network.


```python
import os
import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
from PIL import Image
```


## Data loading and preprocessing

Again we use the cats and dogs dataset:


```python
!wget -nc --show-progress --no-check-certificate https://download.microsoft.com/download/3/e/1/3e1c3f21-ecdb-4869-8368-6deba77b919f/kagglecatsanddogs_5340.zip
!unzip -q -n kagglecatsanddogs_5340.zip
!ls
```


Skip corrupted images (there are quite a few in this dataset):


```python
def is_clean_jpeg(filepath):
    try:
        with open(filepath, 'rb') as f:
            # Check JPEG header
            header = f.read(2)
            if header != b'\xff\xd8':
                return False
            
            # Try to fully load the image
            f.seek(0)
            img = Image.open(f)
            img.load()
            
            # re-open and verify
            f.seek(0)
            img2 = Image.open(f)
            img2.verify()
            
        return True
    except Exception:
        return False
        
data_dir = "PetImages"
num_corrupted = 0

for root, dirs, files in os.walk(data_dir):
    for filename in files:
        filepath = os.path.join(root, filename)
        if not is_clean_jpeg(filepath):
            num_corrupted += 1
            os.remove(filepath)
            
print(f"Removed {num_corrupted} corrupted files")
```


Load into a TensorFlow dataset, using the Keras utility functions.

If the training is going too slow, you can optionally reduce the dimensions of the images (currently set to 180x180 pixels), and adjust the batch size.


```python
image_shape = (180, 180, 3) # TODO reduce if needed
batch_size = 128

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="both",
    seed=123,
    shuffle=True,
    image_size=image_shape[:2],
    batch_size=batch_size,
)
```


Pick some example image and show them.

Note that since we set `shuffle=True` in the code cell above, you will see a different image each time you run the cell below. To have the same images each time you can specify `shuffle=False` and have reproducible outputs. For training, however, it's typically better to shuffle the input data.


```python
# Select one single batch from the dataset
batch = train_ds.take(1)

plt.figure()

for images, labels in batch:
    for i in range(3):
        ax = plt.subplot(1, 3, i+1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
```


## Adding augmentations

In Keras, different types of image augmentations are implemented as layers. This means that once instantiated, they can be used as functions that take in an image and gives a transformed image back. In addition, they can be added as part of a model, just like any other kinds of layers.

**Note:** When adding augmentation layers to a model, they should only be active during training, and not during evaluation and inference -- since we don't want to tamper with new images that our finished model is trying to classify. Keras disables the augmentation layers automatically when we run `model.predict()` or `model.evaluate()`.


```python
plt.figure()
for images, labels in batch:

    images = keras.layers.RandomTranslation(0.2, 0.2)(images)

    for i in range(3):
        ax = plt.subplot(1, 3, i+1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
```


### <span style="color: red; font-weight: bold;">Exercise:<span>

Make the same plots as above, but for all the available augmentation techniques in https://keras.io/api/layers/preprocessing_layers/image_augmentation/.

Put them in a nice layout so that you can compare the effects for each type.

_Hint:_ In case you find it useful to add the augmentation layers in a list and iterate through it, the first element can be a `keras.layers.Identity()` layer, which does nothing except return the original image.


```python
# Your code here
```


## Train some models

Now it is time to put our augmentations to the test.


### Baseline model

For a comparison, let's first train a model with **no** augmentation, on the **full** training dataset (18 728) images.


```python
baseline_model = keras.Sequential(
    [
        keras.Input(shape=image_shape),
        keras.layers.Rescaling(1.0/255),    # Standardise the images
        keras.layers.Conv2D(64, 3, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D(3, padding='same'),
        keras.layers.Conv2D(64, 3, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D(3, padding='same'),
        keras.layers.Conv2D(64, 3, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

baseline_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```


```python
reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, min_lr=0.0001, verbose=1)
early_stop = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

baseline_model.fit(
    train_ds,
    epochs=20,
    validation_data=val_ds,
    callbacks=[early_stop, reduce_lr]
)
```


The final evaluation of the baseline model:


```python
baseline_result = baseline_model.evaluate(val_ds, verbose=0)
print()
print("Accuracy of the baseline model was {}%".format(baseline_result[1]*100))
```


## Train on augmented data

Now for the challenge: We **remove** images from the training set, and our task is to match (or maybe even exceed?) the performance of the baseline model.

Let's make the training dataset 2/3 the size.


```python
reduced_train_ds = train_ds.take((2*len(train_ds))//3)
print('train_ds contains', len(train_ds), 'batches (of 128 images each)')
print('reduced_train_ds contains', len(reduced_train_ds), 'batches')
```


### <span style="color: red; font-weight: bold;">Exercise:<span>

Now, add your favourite augmentation layers to the model:


```python
augmented_model = keras.Sequential(
    [
        # TODO
        # Add augmentation
        keras.Input(shape=image_shape),
        keras.layers.Rescaling(1.0/255),
        keras.layers.Conv2D(64, 3, kernel_initializer='he_uniform', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D(3, padding='same'),
        keras.layers.Conv2D(64, 3, kernel_initializer='he_uniform', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D(3, padding='same'),
        keras.layers.Conv2D(64, 3, kernel_initializer='he_uniform', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

augmented_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, min_lr=0.0001, verbose=1)
early_stop = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

augmented_model.fit(
    reduced_train_ds,   # use the correct dataset
    epochs=20,
    validation_data=val_ds,
    callbacks=[reduce_lr, early_stop]
)
```


```python
augmented_result = augmented_model.evaluate(val_ds, verbose=0)
print('Accuracy of the augmented model was {}%'.format(augmented_result[1]*100))

if augmented_result[1] > baseline_result[1]:
    print()
    print('You\'re awesome!')
```


For the finale, train the augmented model on the full dataset!


```python
# Your code here
```


```python
full_result = augmented_model_full_dataset.evaluate(val_ds)
print('Accuracy of the augmented model in full dataset was {}%'.format(full_result[1]*100))
```

