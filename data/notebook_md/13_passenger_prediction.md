<!-- source: 13_passenger_prediction.ipynb -->

## Predict rail and bus travellers using RNNs

In this notebook we'll try to predict the number of passengers on bus and railroad transport in Chicago, by looking at historical time series data. This task is also described in the "Hands-on Machine Learning" book by A. Geron (that we used in DAT158), so you can also study chapter 15 there for more details.


```python
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
```


## Download the data


```python
filepath = keras.utils.get_file(
    "ridership.tgz",
    "https://github.com/ageron/data/raw/main/ridership.tgz",
    cache_dir=".",
    extract=True
)
if "_extracted" in filepath:
    ridership_path = Path(filepath) / "ridership"
else:
    ridership_path = Path(filepath).with_name("ridership")
```


```python
! ls datasets/ridership_extracted/ridership/
```


Have a quick look at the contents of the CSV file.


```python
! head datasets/ridership_extracted/ridership/CTA_-_Ridership_-_Daily_Boarding_Totals.csv
```


Since we will be dealing with CSV data both in this notebook an the next one, we can use the Pandas library to get some convenience functions.


```python
! pip install pandas
```


Read the file.


```python
import pandas as pd

path = Path("datasets/ridership_extracted/ridership/CTA_-_Ridership_-_Daily_Boarding_Totals.csv")
df = pd.read_csv(path, parse_dates=["service_date"])
df.columns = ["date", "day_type", "bus", "rail", "total"]  # shorter names
df = df.sort_values("date").set_index("date")
df = df.drop("total", axis=1)
df = df.drop_duplicates()  # remove duplicated months (2011-10 and 2014-07)
```


Check if this looks like what we printed before:


```python
df.head()
```


## Investigate the data

Let's look at the first few months of 2019 (note that Pandas treats the range boundaries as inclusive):


```python
df["2019-03":"2019-04"].plot(grid=True, marker=".", figsize=(8, 5))
plt.show()
```


Now, let's look at the difference between each time step and the same day last week. We can compute this by using the `diff` function.

Does it look like there is always the same number of travellers each Tuesday?


```python
# `.diff(7)` means we use a 7 day lag
diff_7 = df[["bus", "rail"]].diff(7)["2019-03":"2019-05"]

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 5))

df.plot(ax=axs[0], legend=False, marker=".")  # original time series
df.shift(7).plot(ax=axs[0], grid=True, legend=False, linestyle=":")  # lagged
diff_7.plot(ax=axs[1], grid=True, marker=".")  # 7-day difference time series

axs[0].set_ylim([170_000, 900_000])
plt.show()
```


```python
list(df.loc["2019-05-25":"2019-05-27"]["day_type"])
```


## Simple prediction

Is the number of travellers on the same day last week, a good estimate of the number of travellers today? Compute the mean absolute error (MAE).


```python
diff_7.abs().mean()
```


And mean absolute percentage error (MAPE):


```python
targets = df[["bus", "rail"]]["2019-03":"2019-05"]
(diff_7 / targets).abs().mean()
```


Now let's look at the yearly seasonality and the long-term trends:


```python
period = slice("2001", "2019")

df_monthly = df.select_dtypes(include="number").resample('ME').mean()  # compute the mean for each month
rolling_average_12_months = df_monthly.loc[period].rolling(window=12).mean()

fig, ax = plt.subplots(figsize=(8, 4))
df_monthly[period].plot(ax=ax, marker=".")
rolling_average_12_months.plot(ax=ax, grid=True, legend=False)

plt.show()
```


```python
df_monthly.diff(12)[period].plot(grid=True, marker=".", figsize=(8, 3))
plt.show()
```


## Optional: Build an ARIMA model

If you want, try also to compare to a "traditional" statistical model, like [ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) (autoregressive integrated moving average).

For this we need the `statsmodels` library:


```python
! pip install statsmodels
```


```python
from statsmodels.tsa.arima.model import ARIMA

origin, today = "2019-01-01", "2019-05-31"
rail_series = df.loc[origin:today]["rail"].asfreq("D")

model = ARIMA(
    rail_series,
    order=(1, 0, 0),
    seasonal_order=(0, 1, 1, 7)
)

model = model.fit()

y_pred = model.forecast()  # should return 427,758.6
```


```python
y_pred[0]  # ARIMA forecast
```


```python
df["rail"].loc["2019-06-01"]  # target value
```


```python
df["rail"].loc["2019-05-25"]  # naive forecast (value from one week earlier)
```


Compute the mean average error for the ARIMA model:


```python
origin, start_date, end_date = "2019-01-01", "2019-03-01", "2019-05-31"
time_period = pd.date_range(start_date, end_date)
rail_series = df.loc[origin:end_date]["rail"].asfreq("D")

y_preds = []
for today in time_period.shift(-1):
    model = ARIMA(rail_series[origin:today],  # train on data up to "today"
                  order=(1, 0, 0),
                  seasonal_order=(0, 1, 1, 7))
    model = model.fit()  # note that we retrain the model every day!
    y_pred = model.forecast().iloc[0]
    y_preds.append(y_pred)

y_preds = pd.Series(y_preds, index=time_period)

mae = (y_preds - rail_series[time_period]).abs().mean()  # should return 32,040.7
```


```python
print('ARIMA model MAE:', mae)
```


Plot the ARIMA forecast:


```python
fig, ax = plt.subplots(figsize=(8, 3))
rail_series.loc[time_period].plot(label="True", ax=ax, marker=".", grid=True)
ax.plot(y_preds, color="r", marker=".", label="ARIMA forecasts")
plt.legend()
plt.show()
```


#### (end of optional part)


## Using `tf.data.Dataset` with time series

Once we have the data loaded as a NumPy array, we can convert it into a batched TensorFlow dataset using the `timeseries_dataset_from_array` utility function.


First, to get even more feel to how `timeseries_dataset_from_array` works, let's print some values from a "dummy" example:


```python
my_series = [0, 1, 2, 3, 4, 5]
my_dataset = tf.keras.utils.timeseries_dataset_from_array(
    my_series,
    targets=my_series[3:],  # the targets are 3 steps into the future
    sequence_length=3,
    batch_size=2
)

as_list = list(my_dataset)
for x, y in as_list:
    print('x:', x.numpy(), 'y:', y.numpy())
```


Ok. Before we continue looking at the data, let's split the time series into three periods, for training, validation and testing. We won't look at the test data for now.


```python
rail_train = df["rail"]["2016-01":"2018-12"] / 1e6      # here we also scale the data
rail_valid = df["rail"]["2019-01":"2019-05"] / 1e6
rail_test = df["rail"]["2019-06":] / 1e6
```


Make the datasets -- but for now, we stick to only the rail data.


```python
seq_length = 56
tf.random.set_seed(42)  # for reproducibility

train_ds = tf.keras.utils.timeseries_dataset_from_array(
    rail_train.to_numpy(),
    targets=rail_train[seq_length:],
    sequence_length=seq_length,
    batch_size=32,
    shuffle=True,
    seed=42
)

valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    rail_valid.to_numpy(),
    targets=rail_valid[seq_length:],
    sequence_length=seq_length,
    batch_size=32
)
```


## Train a simple dense network

For our first network, we try a dead simple approach with a single `Dense` layer.


```python
model = tf.keras.Sequential(
    [
        keras.layers.Input(shape=(seq_length,)),
        keras.layers.Dense(1)
    ]
)

early_stopping_cb = keras.callbacks.EarlyStopping(
    monitor="val_mae",
    patience=20,
    restore_best_weights=True
)

opt = keras.optimizers.Adam(learning_rate=0.02)

model.compile(
    loss=keras.losses.Huber(),
    optimizer=opt,
    metrics=["mae"]
)

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=200,
    callbacks=[early_stopping_cb]
)
```


Evaluate the prediction preformance:


```python
valid_loss, valid_mae = model.evaluate(valid_ds, verbose=0)
print('Validation MAE:', valid_mae * 1e6)   # (remember to multiply with 1e6 since we scaled the data)
```


## Train a simple RNN


Now we want to compare many different model types. Let's first define a utility function to train and evaluate each model.


```python
def fit_and_evaluate(model, train_set, valid_set, learning_rate, epochs=200):

    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_mae",
        patience=20,
        restore_best_weights=True
    )
    opt = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss=keras.losses.Huber(),
        optimizer=opt,
        metrics=["mae"]
    )

    history = model.fit(
        train_set,
        validation_data=valid_set,
        epochs=epochs,
        callbacks=[early_stopping_cb]
    )

    valid_loss, valid_mae = model.evaluate(valid_set)
    return valid_mae * 1e6
```


Construct the simple RNN model, and evaluate it.


```python
model = tf.keras.Sequential(
    [
        keras.layers.Input(shape=(None, 1)),
        keras.layers.SimpleRNN(1)
    ]
)
```


```python
fit_and_evaluate(model, train_ds, valid_ds, learning_rate=0.02)
```


## Train a second, simple RNN

Let's try again, but adding a `Dense` output layer.


```python
univariate_model = tf.keras.Sequential(
    [
        keras.layers.Input(shape=(None, 1)),
        tf.keras.layers.SimpleRNN(32),
        tf.keras.layers.Dense(1)  # no activation function by default
])
```


```python
fit_and_evaluate(univariate_model, train_ds, valid_ds, learning_rate=0.02)
```


How did this compare?


## Train a deep RNN

### Exercise:

This model won't run, because we have forgotten some arguments to the stacked SimpleRNN layers. Fix them, and run the model.


```python
deep_model = tf.keras.Sequential(
    [
        keras.layers.Input(shape=(None, 1)),
        tf.keras.layers.SimpleRNN(32),
        tf.keras.layers.SimpleRNN(32),
        tf.keras.layers.SimpleRNN(32),
        tf.keras.layers.Dense(1)
    ]
)
```


```python
fit_and_evaluate(deep_model, train_ds, valid_ds, learning_rate=0.01)
```


## Multivariate time series

Since we have additional observables in the dataset (day type and bus travels), we can add those to out model too, and se if the results improve:


```python
df_mulvar = df[["bus", "rail"]] / 1e6  # use both bus & rail series as input
df_mulvar["next_day_type"] = df["day_type"].shift(-1)  # we know tomorrow's type
df_mulvar = pd.get_dummies(df_mulvar, dtype=float)  # one-hot encode the day type
```


```python
mulvar_train = df_mulvar["2016-01":"2018-12"]
mulvar_valid = df_mulvar["2019-01":"2019-05"]
mulvar_test = df_mulvar["2019-06":]
```


```python
train_mulvar_ds = tf.keras.utils.timeseries_dataset_from_array(
    mulvar_train.to_numpy(),  # use all 5 columns as input
    targets=mulvar_train["rail"][seq_length:],  # forecast only the rail series
    sequence_length=seq_length,
    batch_size=32,
    shuffle=True,
    seed=42
)

valid_mulvar_ds = tf.keras.utils.timeseries_dataset_from_array(
    mulvar_valid.to_numpy(),
    targets=mulvar_valid["rail"][seq_length:],
    sequence_length=seq_length,
    batch_size=32
)
```


### Exercise:

Implement a single-layer RNNs as before, which now has the correct input shape to match the new dataset.


```python
multivar_model = tf.keras.Sequential([
    ...
])
```


```python
fit_and_evaluate(
    multivar_model,
    train_mulvar_ds,
    valid_mulvar_ds,
    learning_rate=0.05
)
```


## Adding multiple targets to our dataset

Now we try and predict both the number of rail passengers and bus passengers at the same time.


```python
seq_length = 56

train_multask_ds = tf.keras.utils.timeseries_dataset_from_array(
    mulvar_train.to_numpy(),
    targets=mulvar_train[["bus", "rail"]][seq_length:],  # 2 targets per day
    sequence_length=seq_length,
    batch_size=32,
    shuffle=True,
    seed=42
)

valid_multask_ds = tf.keras.utils.timeseries_dataset_from_array(
    mulvar_valid.to_numpy(),
    targets=mulvar_valid[["bus", "rail"]][seq_length:],
    sequence_length=seq_length,
    batch_size=32
)
```


### Exercise:

Again make a simple RNN network, which now can predict two outputs.


```python
multitask_model = tf.keras.Sequential([
    ...
])

fit_and_evaluate(
    multitask_model,
    train_multask_ds,
    valid_multask_ds,
    learning_rate=0.02
)
```


## Forecasting several steps ahead

Let's try to expand our forecast, by forecasting day by day  for multiple days.


```python
X = rail_valid.to_numpy()[np.newaxis, :seq_length, np.newaxis]
for step_ahead in range(14):
    y_pred_one = univariate_model.predict(X)
    X = np.concatenate([X, y_pred_one.reshape(1, 1, 1)], axis=1)
```


```python
# The forecasts start on 2019-02-26, as it is the 57th day of 2019, and they end
# on 2019-03-11. That's 14 days in total.

Y_pred = pd.Series(
    X[0, -14:, 0],
    index=pd.date_range("2019-02-26", "2019-03-11")
)

fig, ax = plt.subplots(figsize=(8, 3.5))

(rail_valid * 1e6)["2019-02-01":"2019-03-11"].plot(
    label="True", marker=".", ax=ax)

(Y_pred * 1e6).plot(
    label="Predictions", grid=True, marker="x", color="r", ax=ax)

ax.vlines("2019-02-25", 0, 1e6, color="k", linestyle="--", label="Today")
ax.set_ylim([200_000, 800_000])
plt.legend(loc="center left")

plt.show()
```


### Open exercises

Since we already know that the `SimpleRNN` layer is a bit too simplistic, try to make two improved models:
- One using [`LSTM` layers](https://keras.io/api/layers/recurrent_layers/lstm/)
- One using [`GRU` layers](https://keras.io/api/layers/recurrent_layers/gru/)


```python

```

