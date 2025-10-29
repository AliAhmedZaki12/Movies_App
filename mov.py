# app.py

import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ====================================================
# 1. إعداد الصفحة في Streamlit
# ====================================================
st.set_page_config(page_title="🎬 Movie Recommendation System", layout="wide")

# ====================================================
# 2. تحميل مفتاح TMDB من البيئة (آمن)
# ====================================================
load_dotenv()  # تحميل القيم من ملف .env أو من GitHub Secrets
api_key = os.getenv("TMDB_API_KEY")

if not api_key:
    st.error("❌ لم يتم العثور على TMDB_API_KEY! تأكد من إضافته إلى Secrets أو .env")
    st.stop()

# ====================================================
# 3. تحميل البيانات ومعالجتها
# ====================================================
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

# ====================================================
# 4. إنشاء مصفوفة التشابه
# ====================================================
@st.cache_data
def create_similarity_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    similarity = cosine_similarity(tfidf_matrix)
    return similarity

similarity_matrix = create_similarity_matrix(movies)

# ====================================================
# 5. جلب البوستر من TMDB API
# ====================================================
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

# ====================================================
# 6. جلب رابط التريلر من YouTube عبر TMDB
# ====================================================
def fetch_trailer_url(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={api_key}&language=en-US'
    response = requests.get(url)
    data = response.json()
    for video in data.get('results', []):
        if video['type'] == 'Trailer' and video['site'] == 'YouTube':
            return f"https://www.youtube.com/watch?v={video['key']}"
    return None

# ====================================================
# 7. دالة التوصية بالأفلام
# ====================================================
def recommend(movie_title, df, similarity_matrix, top_n=10):
    if movie_title not in df['title'].values:
        return pd.DataFrame()
    idx = df[df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    recommended = df.iloc[[i[0] for i in sim_scores]]
    return recommended

# ====================================================
# 8. واجهة Streamlit
# ====================================================
st.title("🎬 Movie Recommendation System")
st.markdown("Select a movie and get recommendations based on its plot, genres, cast, keywords, and director.")

movie_titles = sorted(movies['title'].dropna().unique())
selected_movie = st.selectbox("🎥 Select a movie:", movie_titles)

if st.button("🔍 Show Recommendations"):
    results = recommend(selected_movie, movies, similarity_matrix)

    if results.empty:
        st.warning("⚠️ No recommendations found for this movie.")
    else:
        st.subheader(f"🎯 Top Movies Similar to: {selected_movie}")
        st.markdown("---")

        # ===== التعديل: عرض الأفلام أفقيًا =====
        num_cols = 5  # عدد الأفلام في الصف الواحد
        cols = st.columns(num_cols)

        for idx, row in enumerate(results.itertuples()):
            col = cols[idx % num_cols]  # توزيع الأفلام على الأعمدة
            with col:
                poster_url, rating = fetch_poster(row.id)
                st.image(poster_url, use_container_width=True)
                st.markdown(f"**🎞️ {row.title}**")
                st.markdown(f"⭐ {row.vote_average:.1f}")

                trailer_url = fetch_trailer_url(row.id)
                if trailer_url:
                    st.markdown(f"[▶️ Watch Trailer]({trailer_url})")

        st.markdown("---")

st.caption("🚀 Developed by Ali Ahmed Zaki")

# تحسينات شكلية بسيطة
st.markdown("""
    <style>
    img {border-radius: 12px;}
    .stImage {margin-bottom: -10px;}
    </style>
""", unsafe_allow_html=True)
