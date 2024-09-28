# House Price Prediction Using Machine Learning

This repository contains a machine learning project to predict house prices using various regression models. The project demonstrates data preprocessing, feature engineering, model training, evaluation, and deployment using Streamlit for an interactive user interface.

## Table of Contents

1. [Overview](#overview)
2. [Data](#data)
3. [Installation](#installation)
4. [Modeling Techniques](#modeling-techniques)
5. [Model Evaluation](#model-evaluation)
6. [Web App Interface](#web-app-interface)
7. [Usage](#usage)
8. [Results](#results)
9. [Contributing](#contributing)
10. [License](#license)

## Overview

This project aims to build a reliable and accurate predictive model to estimate house prices based on various factors such as the number of bedrooms, bathrooms, house condition, and more. The main objective is to explore different regression algorithms and select the best model based on performance metrics like Mean Absolute Percentage Error (MAPE).

Key Highlights:
- Data preprocessing and feature scaling
- Implementation of Linear Regression, Ridge, Lasso, ElasticNet, and Stochastic Gradient Descent (SGD) regressors
- Hyperparameter tuning using GridSearchCV
- Deployment of the model with a Streamlit web application

## Data

The dataset used in this project contains the following features:
- **number of bedrooms**
- **number of bathrooms**
- **number of floors**
- **waterfront present**
- **number of views**
- **condition of the house**
- **grade of the house**
- **Built Year**
- **Renovation Year**
- **living area**
- **lot area**
- **latitude**
- **longitude**
- **Price** (Target variable)

**Data Source:** [house_price_pred.csv](./house_price_pred.csv)

The target variable is the **house price**, which is predicted based on the aforementioned features.

## Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/house_price_prediction.git
   cd house_price_prediction
