import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration
st.set_page_config(page_title="Anime Explorer & Recommender", layout="wide")

# -----------------------------------------------------------------------------
# DATA LOADING & PREPROCESSING
# -----------------------------------------------------------------------------

@st.cache_data
def load_data():
    # Load dataset
    df = pd.read_csv('anime_ratings.xls.csv')
    
    # Clean up column spaces if any
    df.columns = df.columns.str.strip()
    
    # Handle missing or invalid values
    df['genre'] = df['genre'].fillna('Unknown')
    df['type'] = df['type'].fillna('Unknown')
    
    # In this dataset, a rating of -1 means the user watched it but didn't rate it.
    # We replace -1 with NaN to calculate an accurate global average rating.
    df_ratings_clean = df.copy()
    df_ratings_clean['rating'] = df_ratings_clean['rating'].replace(-1, np.nan)
    
    # Aggregate to get unique anime rows with their average ratings
    anime_unique = df_ratings_clean.groupby('anime_id').agg({
        'name': 'first',
        'genre': 'first',
        'type': 'first',
        'episodes': 'first',
        'members': 'first',
        'rating': 'mean' # Calculate average user rating
    }).reset_index()
    
    # Round rating to 2 decimal places
    anime_unique['rating'] = anime_unique['rating'].round(2).fillna(0.0)
    
    return anime_unique

try:
    anime_df = load_data()
except FileNotFoundError:
    st.error("Could not find 'anime_ratings.xls.csv'. Please ensure the file is in the same directory as this script.")
    st.stop()