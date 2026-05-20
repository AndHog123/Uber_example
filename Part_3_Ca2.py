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
# MATRIX COMPENSATIONS & SIMILARITY CACHING
# -----------------------------------------------------------------------------
@st.cache_resource
def compute_content_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['genre'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

@st.cache_resource
def compute_collaborative_matrices():
    # Create User-Item Pivot Matrix
    # We use pivot table with filling 0 for unobserved ratings
    pivot_matrix = clean_ratings_df.pivot_table(index='anime_id', columns='user_id', values='rating').fillna(0)
    
    # Item-to-Item Cosine Similarity Matrix (for Collaborative Item Filtering)
    item_collab_sim = cosine_similarity(pivot_matrix)
    
    # User-to-User Cosine Similarity Matrix (for Personalized User Filtering)
    user_pivot_matrix = clean_ratings_df.pivot_table(index='user_id', columns='anime_id', values='rating').fillna(0)
    user_collab_sim = cosine_similarity(user_pivot_matrix)
    
    return pivot_matrix, item_collab_sim, user_pivot_matrix, user_collab_sim

# Generate all matrices
content_sim = compute_content_similarity(anime_df)
pivot_matrix, item_collab_sim, user_pivot_matrix, user_collab_sim = compute_collaborative_matrices()


# -----------------------------------------------------------------------------
# RECOMMENDATION ENGINES
# -----------------------------------------------------------------------------
def get_content_recommendations(title, df, sim_matrix, num_rec=5):
    try:
        idx = df[df['name'] == title].index[0]
    except IndexError:
        return pd.DataFrame()
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:num_rec]
    anime_indices = [i[0] for i in sim_scores]
    return df.iloc[anime_indices][['name', 'genre', 'type', 'rating', 'episodes']]

def get_item_collaborative_recommendations(title, df, pivot, sim_matrix, num_rec=5):
    # Find anime_id corresponding to the name
    try:
        anime_id = df[df['name'] == title]['anime_id'].values[0]
    except IndexError:
        return pd.DataFrame()
    
    if anime_id not in pivot.index:
        return pd.DataFrame()
    
    # Locate actual position inside pivot matrix
    pivot_idx = pivot.index.get_loc(anime_id)
    sim_scores = list(enumerate(sim_matrix[pivot_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Select top matches (skipping itself)
    sim_scores = [s for s in sim_scores if s[0] != pivot_idx][:num_rec]
    
    rec_anime_ids = [pivot.index[i[0]] for i in sim_scores]
    return df[df['anime_id'].isin(rec_anime_ids)][['name', 'genre', 'type', 'rating', 'episodes']]

def get_user_collaborative_recommendations(user_id, df, user_pivot, user_sim, num_rec=5):
    if user_id not in user_pivot.index:
        return pd.DataFrame()
    
    user_idx = user_pivot.index.get_loc(user_id)
    
    # Get top 10 most similar users
    sim_users = list(enumerate(user_sim[user_idx]))
    sim_users = sorted(sim_users, key=lambda x: x[1], reverse=True)[1:11]
    
    sim_user_indices = [x[0] for x in sim_users]
    sim_user_weights = [x[1] for x in sim_users]
    
    # Find what items the target user hasn't rated yet
    target_user_ratings = user_pivot.iloc[user_idx]
    unrated_anime_ids = target_user_ratings[target_user_ratings == 0].index
    
    # Predict scores for unrated items
    predicted_scores = {}
    for idx, weight in zip(sim_user_indices, sim_user_weights):
        if weight == 0:
            continue
        other_user_ratings = user_pivot.iloc[idx]
        for anime_id in unrated_anime_ids:
            if other_user_ratings[anime_id] > 0:
                predicted_scores[anime_id] = predicted_scores.get(anime_id, 0) + (other_user_ratings[anime_id] * weight)
                
    if not predicted_scores:
        return pd.DataFrame()
        
    sorted_predictions = sorted(predicted_scores.items(), key=lambda x: x[1], reverse=True)[:num_rec]
    rec_anime_ids = [item[0] for item in sorted_predictions]
    
    return df[df['anime_id'].isin(rec_anime_ids)][['name', 'genre', 'type', 'rating', 'episodes']]


# -----------------------------------------------------------------------------
# APP INTERFACE LAYOUT
# -----------------------------------------------------------------------------
st.title("🎬 Anime Recommendation & Exploration System")
st.write("Analyze dataset metadata, or make predictions using Content & Collaborative Filtering.")

tab1, tab2, tab3 = st.tabs([
    "📊 Data Catalog Explorer", 
    "🧠 Content-Based Recommender", 
    "👥 Collaborative Filtering"
])

# --- TAB 1: FILTERS & TABLE ---
with tab1:
    st.header("Filter and Explore Catalog")
    
    all_genres = set()
    anime_df['genre'].str.split(', ').dropna().apply(all_genres.update)
    sorted_genres = sorted(list(all_genres))
    sorted_types = sorted(anime_df['type'].unique())
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        search_name = st.text_input("🔍 Search by Name", "")
    with col2:
        selected_genres = st.multiselect("🏷️ Filter by Genre", sorted_genres)
    with col3:
        selected_types = st.multiselect("📺 Filter by Type", sorted_types)
    with col4:
        min_rating = st.slider("⭐ Minimum Average Rating", 0.0, 10.0, 0.0, 0.5)
        
    filtered_df = anime_df.copy()
    if search_name:
        filtered_df = filtered_df[filtered_df['name'].str.contains(search_name, case=False, na=False)]
    if selected_genres:
        filtered_df = filtered_df[filtered_df['genre'].apply(lambda x: any(g in x for g in selected_genres))]
    if selected_types:
        filtered_df = filtered_df[filtered_df['type'].isin(selected_types)]
    filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
    
    st.subheader(f"Results ({len(filtered_df)} items matching)")
    st.dataframe(filtered_df[['name', 'genre', 'type', 'rating', 'episodes', 'members']], use_container_width=True, hide_index=True)


# --- TAB 2: CONTENT-BASED RECOMMENDER ---
with tab2:
    st.header("Content Profile Recommendation")
    st.caption("Matches attributes like genres and themes directly between titles.")
    
    anime_list = sorted(anime_df['name'].unique())
    selected_anime_cb = st.selectbox("🎯 Choose an Anime:", anime_list, key="cb_box")
    num_rec_cb = st.slider("Recommendations Count", 3, 10, 5, key="cb_slider")
    
    if st.button("✨ Generate Content Predictions", key="cb_button"):
        base_info = anime_df[anime_df['name'] == selected_anime_cb].iloc[0]
        st.markdown(f"**Target Profile:** `{base_info['genre']}` | Type: `{base_info['type']}`")
        
        recs = get_content_recommendations(selected_anime_cb, anime_df, content_sim, num_rec_cb)
        if not recs.empty:
            st.dataframe(recs, use_container_width=True, hide_index=True)
        else:
            st.info("No matching attributes found.")


# --- TAB 3: COLLABORATIVE FILTERING ---
with tab3:
    st.header("Behavioral Matrix Collaborative Engine")
    st.caption("Calculates similarities using community rating matrices instead of text tags.")
    
    cf_mode = st.radio("Choose Collaborative Mode:", ["Item-Based (Similar Titles)", "User-Based (Personalized)"])
    
    if cf_mode == "Item-Based (Similar Titles)":
        selected_anime_cf = st.selectbox("🎯 Target Anime:", sorted(anime_df['name'].unique()), key="cf_item_box")
        num_rec_cf_i = st.slider("Recommendations Count", 3, 10, 5, key="cf_i_slider")
        
        if st.button("👥 Discover Behavioral Matches"):
            recs = get_item_collaborative_recommendations(selected_anime_cf, anime_df, pivot_matrix, item_collab_sim, num_rec_cf_i)
            if not recs.empty:
                st.success(f"Users who rated **{selected_anime_cf}** highly also gave top ratings to:")
                st.dataframe(recs, use_container_width=True, hide_index=True)
            else:
                st.warning("Insufficient community interaction data for this title to compute behavioral links.")
                
    else:
        # User-Based personalized recommendation
        sample_users = sorted(clean_ratings_df['user_id'].unique()[:50]) # Grab sample user ids for UI guidance
        st.write(f"💡 *Sample User IDs present in dataset:* {list(sample_users[:10])}...")
        
        target_user = st.number_input("👤 Enter an Existing User ID:", min_value=1, step=1, value=int(sample_users[0]))
        num_rec_cf_u = st.slider("Recommendations Count", 3, 10, 5, key="cf_u_slider")
        
        if st.button("👤 Generate Tailored User Feed"):
            # Display current items that user liked
            user_history = raw_df[raw_df['user_id'] == target_user].sort_values(by='rating', ascending=False)
            if not user_history.empty:
                st.markdown(f"### History for User `{target_user}` (Highly Rated items):")
                st.dataframe(user_history[['name', 'rating']].head(3), use_container_width=True, hide_index=True)
                
                recs = get_user_collaborative_recommendations(target_user, anime_df, user_pivot_matrix, user_collab_sim, num_rec_cf_u)
                if not recs.empty:
                    st.success(f"Top tailored recommendations based on similar peer-profiles:")
                    st.dataframe(recs, use_container_width=True, hide_index=True)
                else:
                    st.info("The system couldn't find enough peer ratings overlaps to construct recommendations.")
            else:
                st.error("User ID not found in the interaction registry.")