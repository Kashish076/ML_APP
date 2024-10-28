import joblib
import numpy as np
import streamlit as st
import pickle
from sklearn.tree import DecisionTreeRegressor

# Load the model and label encoders
with open('saved_model.pkl', 'rb') as file:
    model = pickle.load(file)
regressor_loaded = model['model']
le_country = model["le_country"]
le_education = model["le_education"]

def show_predict_page():
    st.title("Software Developer Salary Prediction")
    st.write("""### We need some information to predict the salary""")
    
    countries = (
        "United States of America",
        "Germany",
        "United Kingdom of Great Britain and Northern Ireland",
        "India",
        "Canada",
        "France",
        "Brazil",
        "Spain",
        "Netherlands",
        "Australia",
        "Italy",
        "Poland",
        "Sweden",
        "Russian Federation",
        "Switzerland"
    )
    
    education_levels = (
        'Master’s degree',
        'Bachelor’s degree',
        'Associate degree',
        'Secondary school',
        'Professional degree',
        'Primary/elementary school'
    )
    
    # User input
    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education_levels)
    experience = st.slider("Years of Experience", 0, 25, 3)
    
    ok = st.button("Calculate Salary")
    if ok:
        # Prepare input for prediction
        x = np.array([[country, education, experience]])
        
        # Transform categorical inputs using fitted label encoders
        x[:, 0] = le_country.fit_transform(x[:, 0])  # Use transform instead of fit_transform
        x[:, 1] = le_education.fit_transform(x[:, 1])  # Use transform instead of fit_transform
        x = x.astype(float)
        
        # Debug: Print transformed input
        st.write(f"Transformed Input: {x}")
        
        regressor = DecisionTreeRegressor(random_state=0)
        
        # Predict the salary
        salary = regressor_loaded.predict(x)
        
        # Debug: Print predicted salary
        st.write(f"Predicted Salary: {salary[0]}")
        
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")

# Run the app
# if __name__ == "__main__":
#     show_predict_page()
