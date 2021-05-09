# -*- coding: utf-8 -*-
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
#     display_name: tsa_course
#     language: python
#     name: tsa_course
# ---

# # Chapter 4 - Introduction to Autoregressive and Automated Methods for Time Series Forecasting

# +
import datetime as dt
import os
import shutil
import warnings
from collections import UserDict
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common.utils import load_data, mape
from IPython.display import Image

# %matplotlib inline

pd.options.display.float_format = "{:,.2f}".format
np.set_printoptions(precision=2)
warnings.filterwarnings("ignore")
# -

data_dir = "./data"
ts_data_load = load_data(data_dir)[["load"]]
ts_data_load.head()

# ## Lag plot

# +
from pandas.plotting import lag_plot

plt.figure()

lag_plot(ts_data_load)
# -

# ## Autocorrelation plot

# ### Autocorrelation Plot Results from ts_data_load dataset

# +
from pandas.plotting import autocorrelation_plot

plt.figure()

autocorrelation_plot(ts_data_load)
# -

# ### Autocorrelation Plot Results from ts_data_load_subset (First week of August 2014)

# +
ts_data_load = load_data("data/")[["load"]]
ts_data_load.head()

ts_data_load_subset = ts_data_load["2014-08-01":"2014-08-07"]

from pandas.plotting import autocorrelation_plot

plt.figure()

autocorrelation_plot(ts_data_load_subset)
# -

# ### Autocorrelation function (acf) plot on ts_data_load dataset

# +
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(ts_data_load)
pyplot.show()
# -

# ### Autocorrelation function (acf) plot on ts_data_load subset

# +
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(ts_data_load_subset)
pyplot.show()
# -

# ### Partial correlation function (pacf) plot on ts_data_load dataset

# +
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(ts_data_load, lags=20)
pyplot.show()
# -

# ### Partial correlation function (pacf) plot on ts_data_load subset

# +
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(ts_data_load_subset, lags=30)
pyplot.show()
# -

# ## Autoregressive method class in Statsmodels

# %matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr
import seaborn as sns
from statsmodels.tsa.api import acf, graphics, pacf
from statsmodels.tsa.ar_model import AutoReg, ar_select_order

model = AutoReg(ts_data_load['load'], 1)
results = model.fit()
print(results.summary())

# #### Note: AutoReg supports describing the same covariance estimators as OLS. Below, we use cov_type="HC0", which is White’s covariance estimator. While the parameter estimates are the same, all of the quantities that depend on the standard error change.

res = model.fit(cov_type="HC0")
print(res.summary())

sns.set_style("darkgrid")
pd.plotting.register_matplotlib_converters()
sns.mpl.rc("figure", figsize=(16, 6))

fig = res.plot_predict(720, 840)

fig = plt.figure(figsize=(16, 9))

fig = res.plot_diagnostics(fig=fig, lags=20)

# ### Prepare the ts_data_load dataset for forecasting task with AutoReg() function

train_start_dt = "2014-11-01 00:00:00"
test_start_dt = "2014-12-30 00:00:00"

# +
train = ts_data_load.copy()[
    (ts_data_load.index >= train_start_dt) & (ts_data_load.index < test_start_dt)
][["load"]]
test = ts_data_load.copy()[ts_data_load.index >= test_start_dt][["load"]]

print("Training data shape: ", train.shape)
print("Test data shape: ", test.shape)
# -

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train["load"] = scaler.fit_transform(train)
train.head()

test["load"] = scaler.transform(test)
test.head()

HORIZON = 3
print("Forecasting horizon:", HORIZON, "hours")

# +
test_shifted = test.copy()

for t in range(1, HORIZON):
    test_shifted["load+" + str(t)] = test_shifted["load"].shift(-t, freq="H")

test_shifted = test_shifted.dropna(how="any")
test_shifted.head(5)

# +
# %%time
training_window = 720

train_ts = train["load"]
test_ts = test_shifted

history = [x for x in train_ts]
history = history[(-training_window):]

predictions = list()

for t in range(test_ts.shape[0]):
    model = AutoReg(history, 1)
    model_fit = model.fit()
    yhat = model_fit.forecast(steps=HORIZON)
    predictions.append(yhat)
    obs = list(test_ts.iloc[t])
    history.append(obs[0])
    history.pop(0)
    print(test_ts.index[t])
    print(t + 1, ": predicted =", yhat, "expected =", obs)
# -

# ## Autoregressive Integrated Moving Average method in Statsmodels

# +
import datetime as dt
import math
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common.utils import load_data, mape
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

# %matplotlib inline
pd.options.display.float_format = "{:,.2f}".format
np.set_printoptions(precision=2)
warnings.filterwarnings("ignore")
# -

data_dir = "./data"
ts_data_load = load_data(data_dir)[["load"]]
ts_data_load.head(10)

train_start_dt = "2014-11-01 00:00:00"
test_start_dt = "2014-12-30 00:00:00"

# +
train = ts_data_load.copy()[
    (ts_data_load.index >= train_start_dt) & (ts_data_load.index < test_start_dt)
][["load"]]
test = ts_data_load.copy()[ts_data_load.index >= test_start_dt][["load"]]

print("Training data shape: ", train.shape)
print("Test data shape: ", test.shape)
# -

scaler = MinMaxScaler()
train["load"] = scaler.fit_transform(train)
train.head()

test["load"] = scaler.transform(test)
test.head()

HORIZON = 3
print("Forecasting horizon:", HORIZON, "hours")

order = (4, 1, 0)
seasonal_order = (1, 1, 0, 24)

# +
model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
results = model.fit()

print(results.summary())

# +
test_shifted = test.copy()

for t in range(1, HORIZON):
    test_shifted["load+" + str(t)] = test_shifted["load"].shift(-t, freq="H")

test_shifted = test_shifted.dropna(how="any")
test_shifted.head(5)

# +
# %%time
training_window = 720

train_ts = train["load"]
test_ts = test_shifted

history = [x for x in train_ts]
history = history[(-training_window):]

predictions = list()

order = (2, 1, 0)
seasonal_order = (1, 1, 0, 24)

for t in range(test_ts.shape[0]):
    model = SARIMAX(endog=history, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    yhat = model_fit.forecast(steps=HORIZON)
    predictions.append(yhat)
    obs = list(test_ts.iloc[t])
    history.append(obs[0])
    history.pop(0)
    print(test_ts.index[t])
    print(t + 1, ": predicted =", yhat, "expected =", obs)
# -

eval_df = pd.DataFrame(
    predictions, columns=["t+" + str(t) for t in range(1, HORIZON + 1)]
)
eval_df["timestamp"] = test.index[0 : len(test.index) - HORIZON + 1]
eval_df = pd.melt(eval_df, id_vars="timestamp", value_name="prediction", var_name="h")
eval_df["actual"] = np.array(np.transpose(test_ts)).ravel()
eval_df[["prediction", "actual"]] = scaler.inverse_transform(
    eval_df[["prediction", "actual"]]
)
eval_df.head()

if HORIZON > 1:
    eval_df["APE"] = (eval_df["prediction"] - eval_df["actual"]).abs() / eval_df[
        "actual"
    ]
    print(eval_df.groupby("h")["APE"].mean())

print(
    "One-step forecast MAPE: ",
    (
        mape(
            eval_df[eval_df["h"] == "t+1"]["prediction"],
            eval_df[eval_df["h"] == "t+1"]["actual"],
        )
    )
    * 100,
    "%",
)

print(
    "Multi-step forecast MAPE: ",
    mape(eval_df["prediction"], eval_df["actual"]) * 100,
    "%",
)
