# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

#
#  ## Training
#  This script executes a training experiment on Azure ML.
#  Once the data is prepared, you can train a model and see the results on Azure.
#  #### There are several steps to follow:
#  * Configure the workspace
#  * Create an experiment
#  * Create or attach a compute cluster
#  * Upload the data to Azure
#  * Create an estimator
#  * Submit the work to the remote cluster
#  * Register the model

#  Import Azure Machine Learning Python SDK and other modules.

# +
import datetime as dt
import math
import os
import urllib.request
import warnings

import azureml.core
import azureml.dataprep as dprep
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from azureml.core import Experiment, Workspace
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.environment import Environment
from azureml.train.estimator import Estimator
from IPython.display import Image, display
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

get_ipython().run_line_magic("matplotlib", "inline")
pd.options.display.float_format = "{:,.2f}".format
np.set_printoptions(precision=2)
warnings.filterwarnings("ignore")  # specify to ignore warning messages
# -

#  ### Configure the workspace
#  Set up your Azure Machine Learning services workspace and configure your notebook library.
#
#  Make sure that you have the correct version of Azure ML SDK.
#  If that's not the case, you can run:
#  * `!pip install --upgrade azureml-sdk[automl,notebooks,explain]`
#  * `!pip install --upgrade azuremlftk`
#
#  Then configure your workspace and write the configuration to a [config.json](https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/machine-learning/service/how-to-configure-environment.md#create-a-workspace-configuration-file) file or read your config.json file to get your workspace.
#  As a second option, one can copy the config file from the Azure workspace in an `.azureml` folder.
#
#  #### In an Azure workspace you will find:
#    * Experiment results
#    * Trained models
#    * Compute targets
#    * Deployment containers
#    * Snapshots
#    * Environments
#    * and more
#
# For more information about the AML services workspace set up, see this [notebook](https://github.com/Azure/MachineLearningNotebooks/blob/master/configuration.ipynb).

print("This notebook was created using version 1.14.0 of the Azure ML SDK")
print("You are currently using version", azureml.core.VERSION, "of the Azure ML SDK")

# +
# # Configure the workspace, if no config file has been downloaded.
# # Give your subscription ID, your ressource group, your workspace_name and your workspace_region

# subscription_id = os.getenv("SUBSCRIPTION_ID", default="d0b8947b-5a39-4d74-944c-48c45b1ccdf3")
# resource_group = os.getenv("RESOURCE_GROUP", default="aml")
# workspace_name = os.getenv("WORKSPACE_NAME", default="timeseries")
# workspace_region = os.getenv("WORKSPACE_REGION", default="centralus")

# try:
#     ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
#     # write the details of the workspace to a configuration file to the notebook library
#     ws.write_config()
#     print("Workspace configuration succeeded")
# except:
#     print("Workspace not accessible. Change your parameters or create a new workspace below")

# +
# Or take the configuration of the existing config.json file

ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep="\n")
# -

# ### Create an environment
#
# We'll create an Azure Machine Learning environment which will help us specify requirements for our model training. This will help us ensure that we use the same versions of libraries such as statsmodels across training and serving
#
# If the environment already exists, then the environment will be overwritten.

my_azureml_env = Environment.from_conda_specification(
    name="my_azureml_env", file_path="./energydemandforecasting/azureml-env.yml"
)
my_azureml_env.register(workspace=ws)

#  ### Create an experiment
#  We’ll create an Azure Machine Learning experiment which will help keep track of the specific data used,
#  as well as the model training job logs.
#
#  If the experiment already exists on the selected workspace, the run will be added to the existing experiment. If not, the experiment will be added to the workspace.

experiment_name = "energydemandforecasting"
exp = Experiment(workspace=ws, name=experiment_name)

#  ### Create or attach an existing compute cluster
#  * For training an ARIMA model, a CPU cluster is enough.
#  *	Note the min_nodes parameter is 0, meaning by default this will have no machines in the cluster and it will automatically scale up and down, so you won't pay for the cluster when you’re not using it.
#  *	You can also enforce policies to control your costs.

# +
# choose a name for your cluster
compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME", "cpucluster")

compute_min_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES", 0)
compute_max_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES", 4)

# This example uses CPU VM. For using GPU VM, set SKU to STANDARD_NC6
vm_size = os.environ.get("AML_COMPUTE_CLUSTER_SKU", "STANDARD_D2_V2")

if compute_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_name]
    if compute_target and type(compute_target) is AmlCompute:
        print("found compute target. just use it. " + compute_name)
else:
    print("creating a new compute target...")
    provisioning_config = AmlCompute.provisioning_configuration(
        vm_size=vm_size, min_nodes=compute_min_nodes, max_nodes=compute_max_nodes
    )

    # create the cluster
    compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)

    # can poll for a minimum number of nodes and for a specific timeout.
    # if no min node count is provided, it will use the scale settings for the cluster
    compute_target.wait_for_completion(
        show_output=True, min_node_count=None, timeout_in_minutes=20
    )

    # For a more detailed view of current AmlCompute status, use 'get_status()'
    print(compute_target.get_status().serialize())
# -

#  ### Upload data to a datastore
#  * Firstly, you can download GEFCom2014 dataset and save the files into a `data` directory locally, which can be done by executing the commented lines in the cell.
#  The data in this example is taken from the GEFCom2014 forecasting competition<sup>1</sup>.
#  It consists of 3 years of hourly electricity load and temperature values between 2012 and 2014.
#
#  * Then, the data is uploaded to the default blob data storage attached to your workspace.
#  The energy file is uploaded into a directory named energy_data at the root of the datastore.
#  The upload of data must be run only the first time. If you run it again, it will skip the uploading of files already present on the datastore.
#
# <sup>1</sup>Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli and Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol.32, no.3, pp 896-913, July-September, 2016.

# +
# data = pd.read_csv("./data/energy.csv")

# # Preview the first 5 lines of the loaded data
# data.head()

# save the files into a data directory locally
data_folder = "./data"

# data_folder = os.path.join(os.getcwd(), 'data')
os.makedirs(data_folder, exist_ok=True)

# import shutil
# from common.utils import extract_data, download_file
# if not os.path.exists(os.path.join(data_folder, 'energy.csv')):
#     # Download and move the zip file
#     download_file("https://mlftsfwp.blob.core.windows.net/mlftsfwp/GEFCom2014.zip")
#     shutil.move("GEFCom2014.zip", os.path.join(data_dir,"GEFCom2014.zip"))
#     # If not done already, extract zipped data and save as csv
#     extract_data(data_dir)
#
# get the default datastore
ds = ws.get_default_datastore()
print(ds.name, ds.datastore_type, ds.account_name, ds.container_name, sep="\n")

# upload the data
ds.upload(
    src_dir=data_folder, target_path="energy_data", overwrite=True, show_progress=True
)

ds = ws.get_default_datastore()
print(ds.datastore_type, ds.account_name, ds.container_name)
# -

#  ### Create an estimator
#  The following parameters will be given to the Estimator:
#  * source directory: the directory which will be uploaded to Azure and contains the script `train.py`.
#  * entry_script: the script that will be executed (train.py).
#  * script_params: the parameters that will be given to the entry script.
#  * compute_target: the the compute cluster that was created above.
#  * conda_dependencies_file: a conda environment yaml specifying the packages in your conda environment, that the script needs.
#
# For more information to define an estimator, see [here](https://docs.microsoft.com/de-ch/python/api/azureml-train-core/azureml.train.estimator.estimator?view=azure-ml-py).

# +
script_params = {
    "--data-folder": ds.path("energy_data").as_mount(),
    "--filename": "energy.csv",
}
script_folder = os.path.join(os.getcwd(), "energydemandforecasting")

est = Estimator(
    source_directory=script_folder,
    script_params=script_params,
    compute_target=compute_target,
    entry_script="train.py",
    conda_dependencies_file="azureml-env.yml",
)
# -

#  ### Submit the job to the cluster

# +
run = exp.submit(config=est)

# specify show_output to True for a verbose log
run.wait_for_completion(show_output=False)
# -

#  ### Register model
#
#  As a last step, we register the model in the workspace, which saves it under 'Models' on Azure, so that you and other collaborators can later query, examine, and deploy this model.
#
#  `outputs` is a directory in your Azure experiment in which the trained model is automatically saved while running the experiment.
#  By registering the model, it is now available on your workspace.

# +
# see files associated with that run
print(run.get_file_names())

# register model
model = run.register_model(model_name="arimamodel", model_path="outputs/arimamodel.pkl")
