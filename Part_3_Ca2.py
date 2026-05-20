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
    
    # Replace -1 (watched but not rated) with NaN to get precise global average ratings
    df_ratings_clean = df.copy()
    df_ratings_clean['rating'] = df_ratings_clean['rating'].replace(-1, np.nan)
    
    # Aggregate to get unique anime rows with their average ratings
    anime_unique = df_ratings_clean.groupby('anime_id').agg({
        'name': 'first',
        'genre': 'first',
        'type': 'first',
        'episodes': 'first',
        'members': 'first',
        'rating': 'mean' 
    }).reset_index()
    
    anime_unique['rating'] = anime_unique['rating'].round(2).fillna(0.0)
    
    return df, df_ratings_clean, anime_unique

try:
    raw_df, clean_ratings_df, anime_df = load_data()
except FileNotFoundError:
    st.error("Could not find 'anime_ratings.xls.csv'. Please ensure the file is in the same directory as this script.")
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
    # One single pivot table: rows = anime_id, columns = user_id 
    pivot_matrix = clean_ratings_df.pivot_table(index='anime_id', columns='user_id', values='rating').fillna(0)
    
    # Item-to-Item Cosine Similarity Matrix
    item_collab_sim = cosine_similarity(pivot_matrix)
    
    return pivot_matrix, item_collab_sim

# Generate optimized matrices
content_sim = compute_content_similarity(anime_df)
pivot_matrix, item_collab_sim = compute_collaborative_matrices()


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
    try:
        anime_id = df[df['name'] == title]['anime_id'].values[0]
    except IndexError:
        return pd.DataFrame()
    
    if anime_id not in pivot.index:
        return pd.DataFrame()
    
    pivot_idx = pivot.index.get_loc(anime_id)
    sim_scores = list(enumerate(sim_matrix[pivot_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != pivot_idx][:num_rec]
    
    rec_anime_ids = [pivot.index[i[0]] for i in sim_scores]
    return df[df['anime_id'].isin(rec_anime_ids)][['name', 'genre', 'type', 'rating', 'episodes']]

def get_user_collaborative_recommendations(user_id, df, pivot, num_rec=5):
    if user_id not in pivot.columns:
        return pd.DataFrame()
    
    # Compute similarity vector for this specific user 
    target_user_vector = pivot[user_id].values.reshape(1, -1)
    user_sim_vector = cosine_similarity(target_user_vector, pivot.T).flatten()
    
    # Get top 10 most similar users
    user_idx = pivot.columns.get_loc(user_id)
    sim_users_indices = np.argsort(user_sim_vector)[::-1]
    sim_users_indices = [idx for idx in sim_users_indices if idx != user_idx][:10]
    
    if len(sim_users_indices) == 0:
        return pd.DataFrame()
        
    similar_users_ids = pivot.columns[sim_users_indices]
    weights = user_sim_vector[sim_users_indices]
    
    # Predict scores for unrated items using matrix multiplication
    user_ratings = pivot[user_id]
    sub_matrix = pivot[similar_users_ids].values
    predicted_scores = np.dot(sub_matrix, weights)
    
    # Mask out already rated items
    predicted_scores[user_ratings > 0] = 0
    
    # Extract top N recommendations
    top_anime_indices = np.argsort(predicted_scores)[::-1]
    top_anime_indices = [idx for idx in top_anime_indices if predicted_scores[idx] > 0][:num_rec]
    
    if not top_anime_indices:
        return pd.DataFrame()
        
    rec_anime_ids = pivot.index[top_anime_indices]
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
        sample_users = sorted(clean_ratings_df['user_id'].unique()[:50])
        st.write(f"💡 *Sample User IDs present in dataset:* {list(sample_users[:10])}...")
        
        target_user = st.number_input("👤 Enter an Existing User ID:", min_value=1, step=1, value=int(sample_users[0]))
        num_rec_cf_u = st.slider("Recommendations Count", 3, 10, 5, key="cf_u_slider")
        
        if st.button("👤 Generate Tailored User Feed"):
            user_history = raw_df[raw_df['user_id'] == target_user].sort_values(by='rating', ascending=False)
            if not user_history.empty:
                st.markdown(f"### History for User `{target_user}` (Highly Rated items):")
                st.dataframe(user_history[['name', 'rating']].head(3), use_container_width=True, hide_index=True)
                
                # Using the newly optimized on-the-fly calculation
                recs = get_user_collaborative_recommendations(target_user, anime_df, pivot_matrix, num_rec_cf_u)
                if not recs.empty:
                    st.success(f"Top tailored recommendations based on similar peer-profiles:")
                    st.dataframe(recs, use_container_width=True, hide_index=True)
                else:
                    st.info("The system couldn't find enough peer ratings overlaps to construct recommendations.")
            else:
                st.error("User ID not found in the interaction registry.")