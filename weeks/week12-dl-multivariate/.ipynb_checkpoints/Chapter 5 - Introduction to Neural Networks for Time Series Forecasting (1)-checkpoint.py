# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernel_info:
#     name: python3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Chapter 5 - Introduction to Neural Networks for Time Series Forecasting

# +
import datetime as dt
import os
import warnings
from collections import UserDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common.utils import load_data, mape
from IPython.display import Image
from sklearn.preprocessing import MinMaxScaler

# %matplotlib inline
# -

pd.options.display.float_format = "{:,.2f}".format
np.set_printoptions(precision=2)
warnings.filterwarnings("ignore")

# !pwd

data_dir = "../../data/"
ts_data_load = load_data(data_dir)[["load"]]
ts_data_load.head()

valid_st_data_load = "2014-09-01 00:00:00"
test_st_data_load = "2014-11-01 00:00:00"

ts_data_load[ts_data_load.index < valid_st_data_load][["load"]].rename(columns={"load": "train"}).join(
    ts_data_load[(ts_data_load.index >= valid_st_data_load) & (ts_data_load.index < test_st_data_load)][
        ["load"]
    ].rename(columns={"load": "validation"}),
    how="outer",
).join(
    ts_data_load[test_st_data_load:][["load"]].rename(columns={"load": "test"}), how="outer"
).plot(
    y=["train", "validation", "test"], figsize=(15, 8), fontsize=12
)
plt.xlabel("timestamp", fontsize=12)
plt.ylabel("load", fontsize=12)
plt.show()

# +
T = 6
HORIZON = 1
train = ts_data_load.copy()[ts_data_load.index < valid_st_data_load][["load"]]

scaler = MinMaxScaler()
train["load"] = scaler.fit_transform(train)

train_shifted = train.copy()
train_shifted["y_t+1"] = train_shifted["load"].shift(-1, freq="H")
for t in range(1, T + 1):
    train_shifted[str(T - t)] = train_shifted["load"].shift(T - t, freq="H")
y_col = "y_t+1"
X_cols = ["load_t-5", "load_t-4", "load_t-3", "load_t-2", "load_t-1", "load_t"]
train_shifted.columns = ["load_original"] + [y_col] + X_cols

train_shifted = train_shifted.dropna(how="any")
train_shifted.head(5)
# -

y_train = train_shifted[y_col].to_numpy()
X_train = train_shifted[X_cols].to_numpy()
X_train = X_train.reshape(X_train.shape[0], T, 1)
y_train.shape

y_train[:3]

X_train.shape

X_train[:3]

train_shifted.head(3)

# +
look_back_dt = dt.datetime.strptime(valid_st_data_load, "%Y-%m-%d %H:%M:%S") - dt.timedelta(
    hours=T - 1
)
valid = ts_data_load.copy()[(ts_data_load.index >= look_back_dt) & (ts_data_load.index < test_st_data_load)][
    ["load"]
]

valid["load"] = scaler.transform(valid)

valid_shifted = valid.copy()
valid_shifted["y+1"] = valid_shifted["load"].shift(-1, freq="H")
for t in range(1, T + 1):
    valid_shifted["load_t-" + str(T - t)] = valid_shifted["load"].shift(T - t, freq="H")

valid_shifted = valid_shifted.dropna(how="any")
valid_shifted.head(3)
# -

y_valid = valid_shifted["y+1"].as_matrix()
X_valid = valid_shifted[["load_t-" + str(T - t) for t in range(1, T + 1)]].as_matrix()
X_valid = X_valid.reshape(X_valid.shape[0], T, 1)

y_valid.shape

y_valid[:3]

X_valid.shape

X_valid[:3]

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.models import Model, Sequential

LATENT_DIM = 5
BATCH_SIZE = 32
EPOCHS = (
    10
)

model = Sequential()
model.add(GRU(LATENT_DIM, input_shape=(T, 1)))
model.add(Dense(HORIZON))

model.compile(optimizer="RMSprop", loss="mse")

model.summary()

earlystop = EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

model_history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_valid, y_valid),
    callbacks=[earlystop],
    verbose=1,
)

# +
look_back_dt = dt.datetime.strptime(test_st_data_load, "%Y-%m-%d %H:%M:%S") - dt.timedelta(
    hours=T - 1
)
test = ts_data_load.copy()[test_st_data_load:][["load"]]

test["load"] = scaler.transform(test)

test_shifted = test.copy()
test_shifted["y_t+1"] = test_shifted["load"].shift(-1, freq="H")
for t in range(1, T + 1):
    test_shifted["load_t-" + str(T - t)] = test_shifted["load"].shift(T - t, freq="H")

test_shifted = test_shifted.dropna(how="any")

y_test = test_shifted["y_t+1"].as_matrix()
X_test = test_shifted[["load_t-" + str(T - t) for t in range(1, T + 1)]].as_matrix()
X_test = X_test.reshape(X_test.shape[0], T, 1)
# -

y_test.shape

X_test.shape

ts_predictions = model.predict(X_test)
ts_predictions

ev_ts_data = pd.DataFrame(
    ts_predictions, columns=["t+" + str(t) for t in range(1, HORIZON + 1)]
)
ev_ts_data["timestamp"] = test_shifted.index
ev_ts_data = pd.melt(ev_ts_data, id_vars="timestamp", value_name="prediction", var_name="h")
ev_ts_data["actual"] = np.transpose(y_test).ravel()
ev_ts_data[["prediction", "actual"]] = scaler.inverse_transform(
    ev_ts_data[["prediction", "actual"]]
)
ev_ts_data.head()


def mape(ts_predictions, actuals):
    """Mean absolute percentage error"""
    return ((ts_predictions - actuals).abs() / actuals).mean()


mape(ev_ts_data["prediction"], ev_ts_data["actual"])

ev_ts_data[ev_ts_data.timestamp < "2014-11-08"].plot(
    x="timestamp", y=["prediction", "actual"], style=["r", "b"], figsize=(15, 8)
)
plt.xlabel("timestamp", fontsize=12)
plt.ylabel("load", fontsize=12)
plt.show()

# ## Multivariate model

# +
import datetime as dt
import os
import sys
import warnings
from collections import UserDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common.utils import TimeSeriesTensor, create_evaluation_df, load_data, mape
from IPython.display import Image

# %matplotlib inline

pd.options.display.float_format = "{:,.2f}".format
np.set_printoptions(precision=2)
warnings.filterwarnings("ignore")
# -
data_dir = "../../data/"
ts_data = load_data(data_dir)
ts_data.head()

valid_st_data_load = "2014-09-01 00:00:00"
test_st_data_load = "2014-11-01 00:00:00"

T = 6
HORIZON = 1
from sklearn.preprocessing import MinMaxScaler
y_scaler = MinMaxScaler()
y_scaler.fit(train[["load"]])

train = ts_data.copy()[ts_data.index < valid_st_data_load][["load", "temp"]]
X_scaler = MinMaxScaler()
train[["load", "temp"]] = X_scaler.fit_transform(train)

train.head()

tensor_structure = {"X": (range(-T + 1, 1), ["load", "temp"])}
ts_train_inp = TimeSeriesTensor(
    dataset=train,
    target="load",
    H=HORIZON,
    tensor_structure=tensor_structure,
    freq="H",
    drop_incomplete=True,
)
back_ts_data = dt.datetime.strptime(valid_st_data_load, "%Y-%m-%d %H:%M:%S") - dt.timedelta(
    hours=T - 1
)
valid = ts_data.copy()[(ts_data.index >= back_ts_data) & (ts_data.index < test_st_data_load)][
    ["load", "temp"]
]
valid[["load", "temp"]] = X_scaler.transform(valid)
valid_inputs = TimeSeriesTensor(valid, "load", HORIZON, tensor_structure)

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.models import Model, Sequential

# +
LATENT_DIM = 5
BATCH_SIZE = 32
EPOCHS = 50

model = Sequential()
model.add(GRU(LATENT_DIM, input_shape=(T, 2)))
model.add(Dense(HORIZON))

# +
model.compile(optimizer="RMSprop", loss="mse")

model.summary()
# -

earlystop = EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

model_history = model.fit(
    ts_train_inp["X"],
    ts_train_inp["target"],
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(valid_inputs["X"], valid_inputs["target"]),
    callbacks=[earlystop],
    verbose=1,
)

# +
back_ts_data = dt.datetime.strptime(test_st_data_load, "%Y-%m-%d %H:%M:%S") - dt.timedelta(
    hours=T - 1
)
ts_data_test = ts_data.copy()[test_st_data_load:][["load", "temp"]]
ts_data_test[["load", "temp"]] = X_scaler.transform(ts_data_test)
ts_data_test_inputs = TimeSeriesTensor(ts_data_test, "load", HORIZON, tensor_structure)

ts_predictions = model.predict(ts_data_test_inputs["X"])

ev_ts_data = create_evaluation_df(ts_predictions, ts_data_test_inputs, HORIZON, y_scaler)
ev_ts_data.head()
# -

mape(ev_ts_data["prediction"], ev_ts_data["actual"])


