![UTA-DataScience-Logo](https://user-images.githubusercontent.com/112208238/235835653-b29143a5-301e-4943-be96-97148cd62bda.png)


# Twitter Sentiment Analysis

* **One Sentence Summary** Ex: This repository holds an attempt to apply LSTMs to Stock Market using data from
"Get Rich" Kaggle challenge (provide link). 

This repository hoolds an attempt to apply the RandomForestClassifier model to the dataset provided by Kaggle in one of it's challenge named 'Twitter sentiment Analysis'.

## Overview

* This section could contain a short paragraph which include the following:
  * The task or challenge presented by the Twitter Entity Sentiment Analysis dataset on Kaggle is to build a machine learning model that can predict the sentiment of tweets towards different entities mentioned in the tweet, such as people, organizations, or locations. The sentiment can be either positive, negative, or neutral. The dataset provides labeled examples of tweets mentioning various entities, along with their corresponding sentiment labels, for training and evaluation purposes. The goal is to create a model that can accurately predict the sentiment of tweets towards different entities in new, unseen data.
  * My approach here is to use the RandomForestClassifier to improve the accuracy so that my model can more accurately tell if the tweet is **Positive**, **Negative**, **Neutral** or **Irrelevant**. 
  * So far the accuracy of my model has been **92%**.

### Data

* Data:
  * Type: For example
    * Input: medical images (1000x1000 pixel jpegs), CSV file: image filename -> diagnosis
    * Input: CSV file of features, output: signal/background flag in 1st column.
  * Size: The training set csv was 10MB.
  * Instances: The training set has 74682 data points which were split in 80-20 format for training the model and testing it upon the remaining tweets for sentiment analysis and determine the accuracy. The archive file also included another csv file with 1000 data points for validation. 


#### Preprocessing / Clean up

* Describe any manipulations you performed to the data.

#### Data Visualization

Show a few visualization of the data and say a few words about what you see.

### Problem Formulation

* Define:
  * Input / Output
  * Models
    * I have used 2 other models for this sentiment analysis Hugging Face and VADER. After examination I felt that the RandomForestClassifier was the easiest to work with and gave the best results.
  * Loss, Optimizer, other Hyperparameters.

### Training

  * Training took near about 20 mins. 

### Conclusions

  * I would say my code isn't good yet but I would work on it to perfect it in the future.

### Future Work

  * My future work would include looking around for more models for better accuracy and try to build my own model for a senti

### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: various functions that are used in cleaning and visualizing data.
  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * models.py: Contains functions that build the various models.
  * training-model-1.ipynb: Trains the first model and saves model during training.
  * training-model-2.ipynb: Trains the second model and saves model during training.
  * training-model-3.ipynb: Trains the third model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.

* Note that all of these notebooks should contain enough text for someone to understand what is happening.

### Software Setup
* List all of the required packages.
* If not standard, provide or point to instruction for installing the packages.
* Describe how to install your package.

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* Provide any references.
