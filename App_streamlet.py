import streamlit as st
import pandas as pd
from joblib import load

# Load a model along with its preprocessor from a given path
def load_model_with_preprocessor(path):
    model_pipeline = load(path)
    return model_pipeline

# Create a Streamlit sidebar for model selection
model_choice = st.sidebar.selectbox(
    'Choose a Model', 
    ('Random Forest', 'Gradient Boosting', 'Support Vector Machine')
)

model_paths = {
    'Random Forest': r'C:\Users\hp\OneDrive\Desktop\Faria\Random Forest Regressor.joblib',
    'Gradient Boosting': r'C:\Users\hp\OneDrive\Desktop\Faria\Gradient Boosting Regressor.joblib',
    'Support Vector Machine': r'C:\Users\hp\OneDrive\Desktop\Faria\Support Vector Machine Regressor.joblib'
}

# Load the selected model along with its preprocessor
loaded_model_pipeline = load_model_with_preprocessor(model_paths[model_choice])

# UI for input features
st.title('ROI Prediction App')
tech_spend = st.number_input('Tech Spend USD', value=10000)
sales_revenue = st.number_input('Sales Revenue USD', value=5e6)
social = st.number_input('Social', value=2500)
employees = st.number_input('Employees', value=100)
vertical = st.selectbox('Vertical', ['Technology And Computing', 'Healthcare', 'Finance', 'Education'])
city = st.selectbox('City', ['San Mateo', 'New York', 'Austin', 'Seattle'])
state = st.selectbox('State', ['CA', 'NY', 'TX', 'WA'])
country = st.selectbox('Country', ['US'])

# Prediction button
if st.button('Predict ROI'):
    # Create DataFrame from input data
    data = pd.DataFrame([[tech_spend, sales_revenue, social, employees, vertical, city, state, country]],
                        columns=['Tech Spend USD', 'Sales Revenue USD', 'Social', 'Employees', 
                                 'Vertical', 'City', 'State', 'Country'])
    
    # Make prediction using the loaded pipeline
    prediction = loaded_model_pipeline.predict(data)
    st.write(f'Predicted ROI: {prediction[0]}')
