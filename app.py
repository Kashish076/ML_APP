import streamlit as st
from predict_page import show_predict_page


st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))
show_predict_page()
