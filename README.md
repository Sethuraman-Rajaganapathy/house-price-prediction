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
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

4. Access the app at [http://localhost:8501](http://localhost:8501) in your browser.

## Modeling Techniques

The following regression techniques were used in this project:

- **Linear Regression**
- **Stochastic Gradient Descent (SGD) Regressor**
- **Ridge Regression** (with GridSearchCV for hyperparameter tuning)
- **Lasso Regression** (with GridSearchCV for hyperparameter tuning)
- **Elastic Net Regression** (with GridSearchCV for hyperparameter tuning)

The models were evaluated based on the following metrics:

- **Mean Absolute Percentage Error (MAPE)**
- **Root Mean Squared Error (RMSE)**
- **R-squared (R²)**
- **Adjusted R-squared**

## Model Evaluation

The best model was selected based on the lowest **MAPE**. The following models were tested, and their performance was recorded in a scorecard for comparison:

- **Linear Regression**
- **SGD Regressor**
- **Ridge Regression (GridSearchCV)**
- **Lasso Regression (GridSearchCV)**
- **Elastic Net Regression (GridSearchCV)**

The model with the lowest MAPE was saved using Python's `pickle` module for deployment.

## Web App Interface

The project includes a user-friendly web interface powered by **Streamlit**, allowing users to input house features (e.g., number of bedrooms, bathrooms, built year, etc.) and get a price prediction in real-time.

### Features:

- Sidebar input sliders for numeric and categorical data
- Dynamic input widgets based on the data type of each feature
- Real-time prediction displayed upon user input

## Usage

Once the app is running, you can interact with it as follows:

1. Adjust the input sliders and dropdowns on the sidebar to specify house features.
2. Click the "Predict House Price" button to get the predicted price based on the selected inputs.
3. The app will display the predicted house price in real-time.

### Example Input

- Bedrooms: 3
- Bathrooms: 2.5
- Year Built: 1995
- Condition: Excellent
- House Grade: 8

### Example Output

```bash
The predicted house price is: $450,000
```
## Contributing

Contributions are welcome! If you'd like to improve this project, feel free to:

1. Fork the repository.
2. Create a new branch: 

    ```bash
    git checkout -b feature-branch
    ```

3. Commit your changes: 

    ```bash
    git commit -m 'Add some feature'
    ```

4. Push to the branch: 

    ```bash
    git push origin feature-branch
    ```

5. Open a Pull Request.
