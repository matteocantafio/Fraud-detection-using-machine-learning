# Fraud Detection with Machine Learning

This project was initially developed for a university course on Data Mining using the **Orange data mining visual programming software**. It has since been reimplemented in Python using Google Colab as a personal project.

## Project Overview

This project aims to develop machine learning models capable of accurately identifying fraudulent credit card transactions. This is crucial to combat increasingly sophisticated fraud techniques and protect users from financial losses. The analysis focuses on imbalanced datasets, employing preprocessing and feature selection techniques to improve the effectiveness of classification models.

### Data

The dataset analyzed contains **284,807 instances** with **30 features** and a binary target class. Two features, *Time* and *Amount*, provide information on the time elapsed since the first transaction and the transaction amount, while the remaining 28 features (V1 to V28) were generated via PCA transformation. The dataset is **highly imbalanced**, with 99.83% of transactions labeled as "Non-Fraudulent" and only 0.17% as "Fraudulent".

### Key Steps (First Approach)

The first approach focuses on the following steps:

*   **Exploratory Data Analysis (EDA):** Initial analysis using tools like Feature Statistics, Distribution, Box Plot, and Scatter Plot. This helped understand the data's characteristics, including identifying that the dataset has no missing values. The average transaction amount is around $88.
*   **Data Preprocessing:** Standardization of *Amount* and *Time* features using z-score scaling to prevent numerical instability.
*   **Feature Selection:** Using a scree plot to visualize variance associated with each feature. The top features were selected which explained 90% of the total variance. Specifically, features from **V1 to V19** were chosen. Correlation analysis ensured independence between the selected features.
*   **Data Balancing:** To address class imbalance, Random Undersampling was applied to balance the dataset. **All 492 fraudulent transactions** were kept and **800 non-fraudulent transactions were randomly selected**. This resulted in a balanced dataset of 1292 instances. This step was done **before the training**.
*   **Model Training:** Three machine learning models were implemented:
    *   Support Vector Machine (SVM)
    *   Neural Network
    *   Decision Tree
*   **Hyperparameter Optimization:** Bayesian Optimization was used to find the optimal hyperparameters for each model. A 5-Fold Cross Validation was performed for each model to maximize the accuracy. The hyperparameters optimized were:
    *   SVM: Regularization parameter C and kernel parameter gamma.
    *   Neural Network: Regularization term alpha and number of neurons in three hidden layers.
    *   Decision Tree: minimum samples in leaf nodes, minimum samples to split a node, and maximum tree depth.
*   **Model Evaluation:** The optimized hyperparameters were used in the respective machine learning models. The performance was evaluated via **Accuracy, F1-Score, and other metrics**

### Software and Libraries

*   **Orange Data Mining** (for initial project development)
*   **Python** (for reimplementation)
*   **Google Colab** (for development environment)
*   Libraries: (The libraries used were not included in the provided sources)
    *   sklearn
    *   pandas
    *   numpy
    *   bayesian-optimization
