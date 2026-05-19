import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration
st.set_page_config(page_title="Anime Explorer & Recommender", layout="wide")

# -----------------------------------------------------------------------------
# DATA LOADING & PREPROCESSING
# -----------------------------------------------------------------------------

@st.cache_data
def load_data():
    # Load dataset
    df = pd.read_csv('anime_ratings.csv')
    
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
    st.error("Could not find 'anime_ratings.csv'. Please ensure the file is in the same directory as this script.")
    st.stop()
    
# -----------------------------------------------------------------------------
# RECOMMENDATION ENGINE ENGINE SETUP
# -----------------------------------------------------------------------------

@st.cache_resource
def compute_similarity(df):
    # Use TF-IDF to vectorize the genres
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['genre'])
    
    # Compute pairwise cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = compute_similarity(anime_df)

def get_recommendations(title, df, cosine_sim, num_recommendations=5):
    # Get index of the anime that matches the title
    try:
        idx = df[df['name'] == title].index[0]
    except IndexError:
        return pd.DataFrame()
    
    # Get pairwise similarity scores of all anime with that anime
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the anime based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get scores of top N most similar anime (excluding itself)
    sim_scores = [score for score in sim_scores if score[0] != idx][:num_recommendations]
    
    # Get the anime indices
    anime_indices = [i[0] for i in sim_scores]
    
    return df.iloc[anime_indices][['name', 'genre', 'type', 'rating', 'episodes']]

# -----------------------------------------------------------------------------
# APP UI LAYOUT
# -----------------------------------------------------------------------------
st.title("🎬 Anime Explorer & Recommendation Dashboard")
st.write("Filter through the dataset catalog or get instant content-based recommendations.")

# Create tabs for clean separation of features
tab1, tab2 = st.tabs(["📊 Data Explorer & Filters", "🤖 Recommendation System"])    

# -----------------------------------------------------------------------------
# TAB 1: DATA EXPLORER WITH ADVANCED FILTERS
# -----------------------------------------------------------------------------
with tab1:
    st.header("Filter and Explore Catalog")
    
    # Extract unique genres for multi-select widget
    all_genres = set()
    anime_df['genre'].str.split(', ').dropna().apply(all_genres.update)
    sorted_genres = sorted(list(all_genres))
    
    # Extract unique types
    sorted_types = sorted(anime_df['type'].unique())
    
    # Setup Filter Grid layout
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        search_name = st.text_input("🔍 Search by Name", "")
        
    with col2:
        selected_genres = st.multiselect("🏷️ Filter by Genre", sorted_genres)
        
    with col3:
        selected_types = st.multiselect("📺 Filter by Type", sorted_types)
        
    with col4:
        min_rating = st.slider("⭐ Minimum Average Rating", 0.0, 10.0, 0.0, 0.5)
        
    # Apply filtering logic iteratively
    filtered_df = anime_df.copy()
    
    if search_name:
        filtered_df = filtered_df[filtered_df['name'].str.contains(search_name, case=False, na=False)]
        
    if selected_genres:
        # Matches rows where any of the selected genres are present
        filtered_df = filtered_df[filtered_df['genre'].apply(lambda x: any(genre in x for genre in selected_genres))]
        
    if selected_types:
        filtered_df = filtered_df[filtered_df['type'].isin(selected_types)]
        
    filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
    
    # Display Results
    st.subheader(f"Results ({len(filtered_df)} matches found)")
    st.dataframe(
        filtered_df[['name', 'genre', 'type', 'rating', 'episodes', 'members']], 
        use_container_width=True,
        hide_index=True
    )