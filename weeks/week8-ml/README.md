# Week 8 `Machine Learning for Time Series`
![image](rand_forest.png)

This week we'll discuss a few important machine learning algorithms used generally for supervised learning problems, but can be used to forecast out time series in particular. We'll be covering how to use Support Vector Machine (SVM) and a decision tree ensemble method called Random Forest (RF).

These are just a start into the many non-deterministic methods for generating a forecasting model for your dataset. Many machine learning algorithmic approaches to forecasting (and modeling in general) are quite different from standard statistical methods in that they often impose far fewer assumptions about the distributions of inputs and apply iterative loss optimization computations to arrive at a best fit. Additionally, this class of methods for time series is typically generalizable to receive mulitvariate inputs.

## Lesson Plan

Read through the markdown, code, and outputs in the following notebooks in order to 
1. [SVM](./les1-svm.ipynb): Develop & evaluate a forecasting model using the SVM algorithm from sklearn
2. [Random Forest](./les2-random_forest.ipynb): Develop & evaluate a forecasting model using the RandomForestRegressor algorithm from sklearn

## Homework

[Bikes](hw1-bikes.ipynb): Problems that ask to apply the above methods to a new dataset

## Solution

[Bikes](sol1-bikes.ipynb): Solution Code

## Course Objectives Addressed

By learning to apply various ML techniques, we've addressed the **6th Course Objective**: Apply machine learning techniques to forecast future values for time series