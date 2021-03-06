# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Project Workflow
The overall workflow/architecture of the project for creating and optimizing the ML pipeline is shown below:

![architecture](architecture.JPG)

## Summary
The dataset contains data about a bank marketing campaign. It contains 21 features related to 322951 potential customers. We need to to predict whether a given client would subscribe to a term deposit or not. The dataset contained information about the last call that was made to the prospective client about the current campaign and the information about the client's credit history and demographic data.

The best performing model was the one produced by the AutoML run. AutoML model's accuracy was 91.59% which was better than the Logistic Regression model with the accuracy of 91.09%.

## Scikit-learn Pipeline

1. The dataset is loaded from the given [URL](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) into the notebook using the `TabularDatasetFactory` class. 
2. The loaded dataset is cleaned using the `clean_data` method written in train.py file. It performs various preprocessing steps, such as: 
   - Dropping null values.
   - One-hot-encoding of categorical features using `get_dummies` function of Pandas or by binarization applying Python's lambda function.
   - Converting/encoding month and day of the week variables from strings to int.
3. Then the data is split into train and test sets in 80:20 ratio using `test_train_split` function of Scikit-Learn.
4. The converted data is then fit into Logistic Regression model.
5. Hyperdrive is used to tune the two hyperparameters called 'C' and 'max-iter'. 
6. The parameter sampler (`RandomParameterSampling`) helps to find the optimal hyperparameters by randomly sampling combinations of them. 
7. The regularization hyperparameter `--C` is the inverse of regularization strength. It is set to ensure that our model does not overfit the data by penalizing addition of features. 
8. The `--max-iter` (maximum iteration) hyperparameter controls the number of iterations to be done before we select our final model (convergence).
9. The bandit termination policy (`BanditPolicy`) helps to stop the iteration early when the primary metric being evaluated is outside the slack factor threshold. It helps to converge to the best model faster and saves time and resources.
10. The optimal values for the hyperparemters are found as `--C` : 0.4601627893840776, `--max_iter`: 100.
11. Using Hyperdrive, scikit-learn's Logistic regression model achieved 91.09% accuracy.

### Hyperparameter Tuning using HyperDrive
The HyperDrive package is used to optimize tuning of hyperparameters by using the `HyperDriveConfig` function. It contains:

- `estimator` (est): A scikit-learn estimator to begin the training and invoke the training script file using the given compute cluster.
- `hyperparameter_sampling`: A `RandomParameterSampling` sampler to randomly select values given in the search space for the two hyper-parameters of Logistic Regression model (`--c` and `--max_iter`).
`policy`: The early termination `BanditPolicy` as mentioned above.
`primary_metric_name`: The primary metric for evaluating the runs - here we use "accuracy" as the primary metric.
`primary_metric_goal`: The goal is to maximize the primary metric "accuracy" (primary_metric_goal.MAXIMIZE) in every run.
`max_concurrent_runs`: Maximum number of runs to run concurrently in the experiment.
`max_total_runs`: Maximum number of total training runs.


## AutoML

In the AutoML pipeline, we first created the tabular dataset with the training data. The data preparation and feature engineering steps are same as scikit learn pipeline.
We can evaluate different models in AutoML. AutoML generated 31 iterations; the best model (VotingEnsemble) comes up with 0.9159 of accuracy with the following hyperparameters selected:

min_samples_split=0.01,
min_weight_fraction_leaf=0.0,
n_estimators=25,
n_jobs=1,
oob_score=True,
random_state=None,
verbose=0,
warm_start=False

## Pipeline comparison
While comparing the two models we can see that AutoML has better performance in accuracy (91.59%) as compared to Logistic Regression (91.09%).
In terms of architecture, AutoML architecture is quite straight forward and user friendly. AutoML ran for 27 options of models and can scale the features of the dataset and cross-validation to prevent bias.

## Future work
We can do statistical analysis first on the data to finalize the features actually significant for the prediction. 
We can check the data imbalance problem in the dataset and can address it with SMOTE library (upsampling or downsampling) or with class_weight hyperparameter of logistic regression. Currently there is data imbalance in the data as the size of the smallest class is 2961 out of 26360 in training data.
We can also try some other hyperparameters such as penalty, solver etc, l1_ration of logistic regression.

## Proof of cluster clean up

cluster.delete() command was used for deleting the cluster:

![cluster_delete_command](./cluster_delete_command.JPG)
![cluster_deleted](./cluster_deleted.JPG)
