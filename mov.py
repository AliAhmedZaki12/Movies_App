# app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# إعداد الصفحة
st.set_page_config(page_title="🎬 Movie Recommendation System", layout="wide")

# ====== 1. تحميل المفتاح من ملف pkl ======
@st.cache_data
def load_api_key():
    with open("tmdb_api_key.pkl", "rb") as f:
        return pickle.load(f)

api_key = load_api_key()

# ====== 2. تحميل البيانات ======
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    df['combined_features'] = (
        df['overview'].fillna('') + ' ' +
        df['genres'].fillna('') + ' ' +
        df['keywords'].fillna('') + ' ' +
        df['cast'].fillna('') + ' ' +
        df['director'].fillna('')
    )
    return df

movies = load_data()

# ====== 3. بناء مصفوفة التشابه ======
@st.cache_data
def create_similarity_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    similarity = cosine_similarity(tfidf_matrix)
    return similarity

similarity_matrix = create_similarity_matrix(movies)

# ====== 4. جلب بوستر الفيلم ======
def fetch_poster(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US'
    response = requests.get(url)
    data = response.json()
    poster_path = data.get('poster_path')
    rating = data.get('vote_average', 'N/A')

    if poster_path:
        poster_url = f'https://image.tmdb.org/t/p/w500{poster_path}'
    else:
        poster_url = 'https://via.placeholder.com/500x750.png?text=No+Poster+Available'

    return poster_url, rating

# ====== 5. جلب رابط التريلر ======
def fetch_trailer_url(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={api_key}&language=en-US'
    response = requests.get(url)
    data = response.json()
    for video in data.get('results', []):
        if video['type'] == 'Trailer' and video['site'] == 'YouTube':
            return f"https://www.youtube.com/watch?v={video['key']}"
    return None

# ====== 6. دالة التوصية ======
def recommend(movie_title, df, similarity_matrix, top_n=5):
    if movie_title not in df['title'].values:
        return pd.DataFrame()
    idx = df[df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    recommended = df.iloc[[i[0] for i in sim_scores]]
    return recommended

# ====== 7. واجهة Streamlit ======
st.title("🎬 Movie Recommendation System")
st.markdown("اختر فيلمًا وسنقترح عليك أفلامًا مشابهة بناءً على القصة، الممثلين، المخرج، والأنواع.")

movie_titles = sorted(movies['title'].dropna().unique())
selected_movie = st.selectbox("🎥 اختر فيلمًا:", movie_titles)

if st.button("🔍 عرض التوصيات"):
    results = recommend(selected_movie, movies, similarity_matrix)

    if results.empty:
        st.warning("⚠️ لا توجد توصيات لهذا الفيلم.")
    else:
        st.subheader(f"✨ أفلام مشابهة لـ: {selected_movie}")
        for i, row in results.iterrows():
            col1, col2 = st.columns([1, 3])
            with col1:
    poster_url, rating = fetch_poster(row['id'])
    st.image(poster_url, use_container_width=True)

            with col2:
                st.markdown(f"### 🎞️ {row['title']}")
                st.markdown(f"⭐️ التقييم: `{row['vote_average']}` | 🗳️ عدد الأصوات: `{row['vote_count']}`")
                st.markdown(f"🎭 الأنواع: `{row['genres']}` | 🎬 المخرج: `{row['director']}`")
                st.markdown(f"📝 {row['overview'][:500]}...")
                trailer_url = fetch_trailer_url(row['id'])
                if trailer_url:
                    st.markdown(f"[▶️ شاهد التريلر على YouTube]({trailer_url})")
            st.markdown("---")

st.caption("🚀 Developed by Ali Ahmed Zaki")

