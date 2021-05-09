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
#     name: python3-azureml
#   kernelspec:
#     display_name: tsa_course
#     language: python
#     name: tsa_course
# ---

# # Chapter 6 -  Model Deployment for Time Series Forecasting - Serving

#  ## Deployment
#  This script allows you to use the model in a webservice and get the desired results.
#  Once the model is trained, it's possible to deploy it in a service.
#  #### For this you need the following steps:
#  * Retrieve the workspace
#  * Get or register the model
#  * Create a docker image
#  * Create the ACI service
#  * Deploy the service
#  * Test the service

#  Import Azure Machine Learning Python SDK and other modules.

# +
import ast
import json
import os

import azureml.core
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from azureml.core import Workspace
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice
from sklearn.preprocessing import MinMaxScaler

from energydemandforecasting.utils import load_data

# -

#  ### Retrieve AML workspace
#  The workspace that was used for training must be retrieved.

ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep="\n")

#  ### Get or register the model (optional)
#  We already registered the model in the training script.
#  But if the model you want to use is only saved locally, you can uncomment and run the following cell, that will register your model in the workspace.
#  Parameters may need adjustment.

# +
# model = Model.register(model_path = "path_of_your_model",
#                        model_name = "name_of_your_model",
#                        tags = {'type': "Time series ARIMA model"},
#                        description = "Time series ARIMA model",
#                        workspace = ws)

# get the already registered model
model = Model.list(ws, name="arimamodel")[0]
print(model)
# -

# ## Get or Register an Environment
#
# We already registered the environment in the training script.

# +
# my_azureml_env = Environment.from_conda_specification(name = "my_azureml_env",
#                                                    file_path = "./energydemandforecasting/azureml-env.yml")
# my_azureml_env.register(workspace=ws)

my_azureml_env = Environment.get(workspace=ws, name="my_azureml_env")

# +
inference_config = InferenceConfig(
    entry_script="energydemandforecasting/score.py", environment=my_azureml_env
)

# Set deployment configuration
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

aci_service_name = "aci-service-arima"

# Define the model, inference, & deployment configuration and web service name and location to deploy
service = Model.deploy(
    workspace=ws,
    name=aci_service_name,
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config,
)

service.wait_for_deployment(True)
# -

#  ### Call the service and test it
#  The service is tested on the `energy.csv` data.

# load the data to use for testing and encode it in json
energy_pd = load_data("./data/energy.csv")
energy = pd.DataFrame.to_json(energy_pd, date_format="iso")
energy = json.loads(energy)
energy = json.dumps({"energy": energy})

# Call the service to get the prediction for this time series
prediction = service.run(energy)

#  ### Plot the result
#  * Convert the prediction to a data frame containing correct indices and columns.
#  * Scale the original data as in the training.
#  * Plot the original data and the prediction.

# +
# prediction is a string, convert it to a dictionary
prediction = ast.literal_eval(prediction)

# convert the dictionary to pandas dataframe
prediction_df = pd.DataFrame.from_dict(prediction)

prediction_df.columns = ["load"]
prediction_df.index = energy_pd.iloc[2500:2510].index

# +
# Scale the original data
scaler = MinMaxScaler()
energy_pd["load"] = scaler.fit_transform(
    np.array(energy_pd.loc[:, "load"].values).reshape(-1, 1)
)

# Visualize a part of the data before the forecasting
original_data = energy_pd.iloc[1500:2501]

# +
# Plot the forecasted data points
fig = plt.figure(figsize=(15, 8))

plt.plot_date(
    x=original_data.index,
    y=original_data,
    fmt="-",
    xdate=True,
    label="original load",
    color="red",
)
plt.plot_date(
    x=prediction_df.index,
    y=prediction_df,
    fmt="-",
    xdate=True,
    label="predicted load",
    color="yellow",
)
# -

# ### Cleanup
# The service costs money during deployment. We should clean this up

service.delete()
