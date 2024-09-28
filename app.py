import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import cross_val_score,KFold,GridSearchCV
from sklearn.preprocessing import StandardScaler as ss
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge,Lasso,ElasticNet
from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error,r2_score

#importing the dataframe
df=pd.read_csv('house_price_pred.csv')
X = df.drop(columns=['Price'])
y=df['Price']

# scaling independent features
scaler=ss()
scaled_x=scaler.fit_transform(X)

# splitting data into test and train data
X_train, X_test, y_train, y_test = tts(scaled_x, y, test_size=0.3, random_state=42)

#defining functions to get a scorecard
def get_rmse(model):
    train_pred=model.predict(X_train)
    mse=mean_squared_error(y_train,train_pred)
    rmse=round(np.sqrt(mse),4)
    return(rmse)

def get_mape(model):
    train_pred=model.predict(X_train)
    mape=round(mean_absolute_percentage_error(y_train,train_pred),3)
    return mape

def get_score(model):
    train_pred=model.predict(X_train)
    r_sq=r2_score(y_train,train_pred)
    n,k=X_train.shape[0],X_train.shape[1]
    adj_r_sq=1-((1-r_sq)*(n-1)/(n-k-1))
    return [r_sq,adj_r_sq]

score_card = pd.DataFrame(columns=['Model_name', 'Alpha', 'L1-ratio', 'R_squared', 'Adjusted_R_sq', 'RMSE', 'MAPE'])

# Function to update score_card
def usc(algorithm_name, model, alpha='-', l1_ratio='-'):
    global score_card
    # Create a new row as a DataFrame
    new_row = pd.DataFrame({
        'Model_name': [algorithm_name],
        'Alpha': [alpha],
        'L1-ratio': [l1_ratio],
        'R_squared': [get_score(model)[0]],
        'Adjusted_R_sq': [get_score(model)[1]],
        'RMSE': [get_rmse(model)],
        'MAPE': [get_mape(model)]
    })
    
    # Concatenate the new row with the existing score_card DataFrame
    score_card = pd.concat([score_card, new_row], ignore_index=True)

# Applying linear regression
linreg = LinearRegression()
MLR_lin = linreg.fit(X_train, y_train)

# Call the usc function to add the Linear Regression model's performance to score_card
usc(algorithm_name='Linear Regression', model=MLR_lin)

sgd=SGDRegressor(random_state=10)
sgd_lin = sgd.fit(X_train, y_train)
usc(algorithm_name='SGD Regressor', model=sgd_lin)

tp=[{'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,0.1,1,5,10,20,40,60,80,100,150,200]}]
ridge=Ridge()
ridge_grid=GridSearchCV(estimator=ridge,param_grid=tp,cv=10)
ridge_grid.fit(X_train,y_train)
usc(algorithm_name='Ridge(Grid Search CV)',model=ridge_grid,alpha=ridge_grid.best_params_.get('alpha'))

tp=[{'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,0.1,1,5,10,20,40,60,80,100,150,200]}]
lasso=Lasso()
lasso_grid=GridSearchCV(estimator=lasso,param_grid=tp,cv=10)
lasso_grid.fit(X_train,y_train)
usc(algorithm_name='Lasso(Grid Search CV)',model=lasso_grid,alpha=lasso_grid.best_params_.get('alpha'))

tp=[{'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,0.1,1,5,10,20,40,60,80,100,150,200],
     'l1_ratio':[0.0001,0.001,0.002,0.01,0.1,0.2]}]
enet=ElasticNet()
enet_grid=GridSearchCV(estimator=enet,param_grid=tp,cv=10)
enet_grid.fit(X_train,y_train)
usc(algorithm_name='Elastic Net Regression(Grid Search CV)',
    model=enet_grid,
    alpha=enet_grid.best_params_.get('alpha'),
    l1_ratio=enet_grid.best_params_.get('l1_ratio'))



# Find the model with the lowest MAPE
best_model_mape = score_card.loc[score_card['MAPE'].idxmin()]
# Save the model with the lowest MAPE as a pickle file
if best_model_mape['Model_name'] == 'Linear Regression':
    model = MLR_lin
elif best_model_mape['Model_name'] == 'SGD Regressor':
    model = sgd_lin
elif best_model_mape['Model_name'] == 'Ridge(Grid Search CV)':
    model = ridge_grid.best_estimator_
elif best_model_mape['Model_name'] == 'Lasso(Grid Search CV)':
    momodeldel_to_save = lasso_grid.best_estimator_
elif best_model_mape['Model_name'] == 'Elastic Net Regression(Grid Search CV)':
    model = enet_grid.best_estimator_

# Save the model to a pickle file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)


# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

st.title("House Price Prediction app")
# Sidebar for user inputs
st.sidebar.header("Enter house details:")

# Dictionary to hold user inputs
user_inputs = {}

# Set specific step sizes and unique values for particular columns
for col in df.columns:
    if col == 'Price':  # Skip 'Price' since it's the target variable
        continue

    if col == 'number of bathrooms' or col == 'number of floors':
        # For these columns, use a step of 0.5
        min_value = df[col].min()
        max_value = df[col].max()
        user_input = st.sidebar.slider(f"{col}", min_value=min_value, max_value=max_value, value=min_value, step=0.5)
        user_inputs[col] = user_input

    elif col == 'Lattitude':
        # Use unique values for latitude
        unique_values = sorted(df[col].unique())
        user_input = st.sidebar.selectbox(f"{col}", unique_values)
        user_inputs[col] = user_input

    else:
        # For other columns, use a step of 1
        min_value = df[col].min()
        max_value = df[col].max()
        user_input = st.sidebar.slider(f"{col}", min_value=int(min_value), max_value=int(max_value), value=int(min_value), step=1)
        user_inputs[col] = user_input

# Convert the user inputs to a DataFrame (for prediction)
user_input_df = pd.DataFrame([user_inputs])

# Display user inputs
st.write("User inputs:", user_input_df)

# When the "Predict House Price" button is clicked
if st.button("Predict House Price"):
    # Scale the input data using the scaler used during training
    input_data_scaled = scaler.transform(user_input_df)

    # Make the prediction
    prediction = model.predict(input_data_scaled)

    # Display the predicted price
    st.write(f"The predicted house price is: ${prediction[0]:,.2f}")

