# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Chapter 4 - Introduction to Autoregressive and Automated Methods for Time Series Forecasting - Azure Machine Learning Example
#
# ## Automated Machine Learning

# +
# This should be done in a seperate environment as azureml-sdk conflicts with some of our package versions such as statsmodels 0.12
# The environment this was tested with is 
# name: azureml
# channels:
#   - defaults
#   - conda-forge
# dependencies:
#   - python=3.6
#   - matplotlib=3.1.1
#   - pandas=1.1.1
#   - pip
#   - pip:
#     - azureml-sdk[automl,notebooks,explain]
#     - azuremlftk
#     - azure-cli

# +
import logging
import os
import warnings
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

import azureml.core
from azureml.core import Dataset, Experiment, Workspace
from azureml.train.automl import AutoMLConfig

warnings.showwarning = lambda *args, **kwargs: None
# -

experiment_name = 'automatedML-timeseriesforecasting'
experiment = Experiment(ws, experiment_name)
output = {}
output['SDK version'] = azureml.core.VERSION
output['Subscription ID'] = ws.subscription_id
output['Workspace'] = ws.name
output['Resource Group'] = ws.resource_group
output['Location'] = ws.location
output['Run History Name'] = experiment_name
pd.set_option('display.max_colwidth', -1)
outputDf = pd.DataFrame(data = output, index = [''])
outputDf.T

# +
cts = ws.compute_targets
if amlcompute_cluster_name in cts and cts[amlcompute_cluster_name].type == "AmlCompute":
    found = True
    print("Found existing compute target.")
    compute_target = cts[amlcompute_cluster_name]

if not found:
    print("Creating a new compute target...")
    provisioning_config = AmlCompute.provisioning_configuration(
        vm_size="STANDARD_DS12_V2",
        max_nodes=6,
    )

    compute_target = ComputeTarget.create(
        ws, amlcompute_cluster_name, provisioning_config
    )

print("Checking cluster status...")

compute_target.wait_for_completion(
    show_output=True, min_node_count=None, timeout_in_minutes=20
)


# +
target_column_name = "demand"
time_column_name = "timeStamp"

ts_data = Dataset.Tabular.from_delimited_files(
    path="https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/nyc_energy.csv"
).with_timestamp_columns(fine_grain_timestamp=time_column_name)

ts_data.take(5).to_pandas_dataframe().reset_index(drop=True)
# -

ts_data = ts_data.time_before(datetime(2017, 10, 10, 5))

# +
train = ts_data.time_before(datetime(2017, 8, 8, 5), include_boundary=True)
train.to_pandas_dataframe().reset_index(drop=True).sort_values(time_column_name).tail(5)

test = ts_data.time_between(datetime(2017, 8, 8, 6), datetime(2017, 8, 10, 5))
test.to_pandas_dataframe().reset_index(drop=True).head(5)
# -

max_horizon = 24

# +
automl_settings = {
    "time_column_name": time_column_name,
    "max_horizon": max_horizon,
}

automl_config = AutoMLConfig(
    task="forecasting",
    primary_metric="normalized_root_mean_squared_error",
    blocked_models=["ExtremeRandomTrees", "AutoArima", "Prophet"],
    experiment_timeout_hours=0.3,
    training_data=train,
    label_column_name=target_column_name,
    compute_target=compute_target,
    enable_early_stopping=True,
    n_cross_validations=3,
    verbosity=logging.INFO,
    **automl_settings
)
# -

remote_run = experiment.submit(automl_config, show_output=False)

remote_run.wait_for_completion()

best_run, fitted_model = remote_run.get_output()
fitted_model.steps

# +
featurization_summary = fitted_model.named_steps[
    "timeseriestransformer"
].get_featurization_summary()

pd.DataFrame.from_records(featurization_summary)
# -

X_test = test.to_pandas_dataframe().reset_index(drop=True)
y_test = X_test.pop(target_column_name).values

y_predictions, X_trans = fitted_model.forecast(X_test)

# +
from common.forecasting_helper import align_outputs

ts_results_all = align_outputs(y_predictions, X_trans, X_test, y_test, target_column_name)

from automl.client.core.common import constants
from azureml.automl.core._vendor.automl.client.core.common import metrics
from matplotlib import pyplot as plt

scores = metrics.compute_metrics_regression(
    ts_results_all["predicted"],
    ts_results_all[target_column_name],
    list(constants.Metric.SCALAR_REGRESSION_SET),
    None,
    None,
    None,
)

print("[Test data scores]\n")
for key, value in scores.items():
    print("{}:   {:.3f}".format(key, value))

# %matplotlib inline
test_pred = plt.scatter(ts_results_all[target_column_name], ts_results_all["predicted"], color="b")
test_test = plt.scatter(
    ts_results_all[target_column_name], ts_results_all[target_column_name], color="g"
)
plt.legend(
    (test_pred, test_test), ("prediction", "truth"), loc="upper left", fontsize=8
)
plt.show()