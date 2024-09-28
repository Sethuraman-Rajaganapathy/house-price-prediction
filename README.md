# House Price Prediction Using Machine Learning
This repository contains a machine learning project to predict house prices using various regression models. The project demonstrates data preprocessing, feature engineering, model training, evaluation, and deployment using Streamlit for an interactive user interface.

Table of Contents
Overview
Data
Installation
Modeling Techniques
Model Evaluation
Web App Interface
Usage
Results
Contributing
License
Overview
This project aims to build a reliable and accurate predictive model to estimate house prices based on various factors such as the number of bedrooms, bathrooms, house condition, and more. The main objective is to explore different regression algorithms and select the best model based on performance metrics like Mean Absolute Percentage Error (MAPE).

Key Highlights:

Data preprocessing and feature scaling
Implementation of Linear Regression, Ridge, Lasso, ElasticNet, and Stochastic Gradient Descent (SGD) regressors
Hyperparameter tuning using GridSearchCV
Deployment of the model with a Streamlit web application
Data
The dataset used in this project contains the following features:

number of bedrooms
number of bathrooms
number of floors
waterfront present
number of views
condition of the house
grade of the house
Built Year
Renovation Year
living area
lot area
latitude
longitude
Price (Target variable)
Data Source: house_price_pred.csv

The target variable is the house price, which is predicted based on the aforementioned features.

Installation
Clone the repository to your local machine:

bash
Copy code
git clone https://github.com/yourusername/house_price_prediction.git
cd house_price_prediction
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit application:

bash
Copy code
streamlit run app.py
Access the app at http://localhost:8501 in your browser.

Modeling Techniques
The following regression techniques were used in this project:

Linear Regression
Stochastic Gradient Descent (SGD) Regressor
Ridge Regression (with GridSearchCV for hyperparameter tuning)
Lasso Regression (with GridSearchCV for hyperparameter tuning)
Elastic Net Regression (with GridSearchCV for hyperparameter tuning)
The models were evaluated based on the following metrics:

Mean Absolute Percentage Error (MAPE)
Root Mean Squared Error (RMSE)
R-squared (R²)
Adjusted R-squared
Model Evaluation
The best model was selected based on the lowest MAPE. The following models were tested, and their performance was recorded in a scorecard for comparison:

Linear Regression
SGD Regressor
Ridge Regression (GridSearchCV)
Lasso Regression (GridSearchCV)
Elastic Net Regression (GridSearchCV)
The model with the lowest MAPE was saved using Python's pickle module for deployment.

Web App Interface
The project includes a user-friendly web interface powered by Streamlit, allowing users to input house features (e.g., number of bedrooms, bathrooms, built year, etc.) and get a price prediction in real-time.

Features:

Sidebar input sliders for numeric and categorical data
Dynamic input widgets based on the data type of each feature
Real-time prediction displayed upon user input
Usage
Once the app is running, you can interact with it as follows:

Adjust the input sliders and dropdowns on the sidebar to specify house features.
Click the "Predict House Price" button to get the predicted price based on the selected inputs.
The app will display the predicted house price in real-time.
Example Input
Bedrooms: 3
Bathrooms: 2.5
Year Built: 1995
Condition: Excellent
House Grade: 8
Example Output
swift
Copy code
The predicted house price is: $450,000
Results
The best-performing model was found to be [Insert Best Model] with the following performance metrics:

Mean Absolute Percentage Error (MAPE): [Insert Value]
Root Mean Squared Error (RMSE): [Insert Value]
R-squared (R²): [Insert Value]
These results demonstrate the model's capability to predict house prices accurately based on the selected features.

Contributing

Contributions are welcome! If you'd like to improve this project, feel free to:

Fork the repository.

Create a new branch: git checkout -b feature-branch.

Commit your changes: git commit -m 'Add some feature'.

Push to the branch: git push origin feature-branch.

Open a Pull Request.

