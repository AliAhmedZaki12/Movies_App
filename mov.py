# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🎬 Movie Recommendation System", layout="wide")

# 1. Load Movies Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")

    # دمج الأعمدة النصية المفيدة للتوصية
    df['combined_features'] = (
        df['overview'].fillna('') + ' ' +
        df['genres'].fillna('') + ' ' +
        df['keywords'].fillna('') + ' ' +
        df['cast'].fillna('') + ' ' +
        df['director'].fillna('')
    )
    return df

movies = load_data()

# 2. Create Similarity Matrix
@st.cache_data
def create_similarity_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    similarity = cosine_similarity(tfidf_matrix)
    return similarity

similarity_matrix = create_similarity_matrix(movies)

# 3. Recommendation Function
def recommend(movie_title, df, similarity_matrix, top_n=10):
    if movie_title not in df['title'].values:
        return pd.DataFrame()
    idx = df[df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    recommended = df.iloc[[i[0] for i in sim_scores]]
    return recommended

# 4. Streamlit UI
st.title("🎬 Movie Recommendation System")
st.markdown("Choose a movie and we'll suggest similar movies based on story, genres, director, actors, and keywords.")

movie_titles = sorted(movies['title'].dropna().unique())
selected_movie = st.selectbox("🎥 Choose a movie:", movie_titles)

if st.button("🔍 Show recommendations "):
    results = recommend(selected_movie, movies, similarity_matrix)

    if results.empty:
        st.warning("⚠️ There are no recommendations for this movie.")
    else:
        st.subheader(f"✨ Similar movies to: {selected_movie}")
        for i, row in results.iterrows():
            st.markdown(f"**🎞️ {row['title']}**")
            st.caption(f"{row['overview'][:400]}..." if len(row['overview']) > 400 else row['overview'])
            st.markdown(f"⭐️ Rating: {row['vote_average']} | 🗳️ Votes: {row['vote_count']}")
            st.markdown(f"🎭 Genres: `{row['genres']}` | 🎬 Director: `{row['director']}`")
            st.markdown("---")

st.caption("Developed by Ali Ahmed Zaki")
