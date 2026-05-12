import streamlit as st
import pandas as pd
import numpy as np

st.title("Uber Pickups in NY")

DATE_COLUMN = 'date/time'
uploaded_files = st.file_uploader(
    "Upload data", accept_multiple_files=True, type="csv"
         
def load_data(nrows):
        data = pd.read_csv(uploaded_files)
        lowercase = lambda x:str(x).lower()
        data.rename(lowercase, axis = "columns", inplace = True)
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
        return data
        
data_load_state = st.text("Loading Data...")
data = load_data(10000)
data_load_state = st.text("Data Loading Completed")

if st.checkbox("Show Raw Data"):
        st.subheader("Raw Data")
        st.write(data)

st.subheader("Num Pickups Per Hour")
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins = 24, range = (0, 24))[0]
st.bar_chart(hist_values)

hour_to_filter = st.slider("hour", 0, 23, 17)
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]


st.subheader("Map of Pickups")
st.map(filtered_data)