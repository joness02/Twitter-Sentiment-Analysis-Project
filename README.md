![UTA-DataScience-Logo](https://user-images.githubusercontent.com/112208238/235835653-b29143a5-301e-4943-be96-97148cd62bda.png)


# Twitter Sentiment Analysis

This repository hoolds an attempt to apply the RandomForestClassifier model to the dataset provided by Kaggle in one of it's challenge [Twitter sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis).

## Overview

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

 * Models
    * I have used 2 other models for this sentiment analysis Hugging Face and VADER. After examination I felt that the RandomForestClassifier was the easiest to work with and gave the best results.

### Training

  * Training took near about 20 mins. 

### Conclusions

  * I would say my code isn't good yet but I would work on it to perfect it in the future.

### Future Work

  * My future work would include looking around for more models for better accuracy and try to build my own model for any sentiment analysis for that instance.

### Overview of files in repository

* The only file in this repository is the **Project-code.ipynb** which provides the code to solve this challenge with accuracy score.

### Data

* The data can be downloaded from the [Kaggle challenge](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis).
