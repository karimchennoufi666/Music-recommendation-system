# spotify_recs_app_fixed.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ================== Spotify Setup ==================
client_id = "f7f30d8beeac46aa9e63c4fb644005ed"
client_secret = "d64f8e38e527458191da4b9e15c155bb"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=client_id,
    client_secret=client_secret
))

# ================== Load your datasets ==================
# Google Drive CSV raw download link
url = "https://drive.google.com/uc?export=download&id=1d38o6a9o5MHzO8JSRR8t8PX8vtrFpI4B"

# Load main dataset
try:
    data = pd.read_csv(url, encoding='utf-8', sep=',', on_bad_lines='skip')
except UnicodeDecodeError:
    data = pd.read_csv(url, encoding='latin1', sep=',', on_bad_lines='skip')

# Load local Spotify tracks dataset
spotify_tracks = pd.read_csv("spotify_tracks.csv")

# Normalize column names
data.columns = data.columns.str.strip().str.lower()
spotify_tracks.columns = spotify_tracks.columns.str.strip().str.lower()

# ================== Functions ==================
def get_song_data(song, dataset):
    """Get song data from dataset or Spotify API"""
    if "name" not in dataset.columns:
        return None

    row = dataset[dataset["name"].str.lower() == song.lower()]
    if not row.empty:
        return row.iloc[0].to_dict()

    try:
        results = sp.search(q=f"track:{song}", type="track", limit=1)
        tracks = results["tracks"]["items"]
        if tracks:
            track = tracks[0]
            return {
                "name": track["name"],
                "artists": [a["name"] for a in track["artists"]],
                "id": track["id"],
                "popularity": track["popularity"],
                "album": track["album"]["name"],
                "release_date": track["album"]["release_date"],
            }
    except Exception as e:
        print(f"Spotify API error: {e}")
        return None

    return None

def get_mean_vector(song_list, dataset, feature_cols):
    """Compute mean vector of features for a list of songs"""
    vectors = []
    for song in song_list:
        data_song = get_song_data(song, dataset)
        if data_song:
            vec = []
            for col in feature_cols:
                value = data_song.get(col)
                if value is None or not isinstance(value, (int, float)):
                    vec.append(0)
                else:
                    vec.append(float(value))
            vectors.append(vec)
    if not vectors:
        return None
    return np.mean(vectors, axis=0)

def recommend_songs(seed_songs, dataset, feature_cols, n=5):
    """Recommend songs based on seed list"""
    mean_vector = get_mean_vector(seed_songs, dataset, feature_cols)
    if mean_vector is None:
        return pd.DataFrame(columns=["name", "artists", "similarity"])

    song_features = dataset[feature_cols].values
    similarities = cosine_similarity([mean_vector], song_features)[0]

    dataset["similarity"] = similarities
    recommendations = dataset.sort_values("similarity", ascending=False)
    recommendations = recommendations[~dataset["name"].str.lower().isin([s.lower() for s in seed_songs])]

    return recommendations.head(n)[["name", "artists", "similarity"]]

# ================== Feature columns ==================
feature_columns = [
    "valence", "danceability", "energy", "loudness",
    "acousticness", "instrumentalness", "liveness",
    "speechiness", "tempo"
]

# ================== Streamlit UI ==================
st.title("🎵 Spotify Song Recommendation System")

# Seed songs input
default_songs = spotify_tracks["name"].tolist()[:5] if "name" in spotify_tracks.columns else []
seed_songs_input = st.text_area(
    "Enter your favorite songs (one per line):",
    value="\n".join(default_songs)
)

# Number of recommendations
num_recs = st.slider("Number of recommendations:", 1, 20, 5)

if st.button("Generate Recommendations"):
    seed_songs_list = [s.strip() for s in seed_songs_input.split("\n") if s.strip()]

    if not seed_songs_list:
        st.warning("Please enter at least one song!")
    else:
        with st.spinner("Generating recommendations..."):
            recommendations = recommend_songs(seed_songs_list, data, feature_columns, n=num_recs)

            if recommendations.empty:
                st.info("No recommendations found.")
            else:
                st.subheader("Recommended Songs:")
                for idx, row in recommendations.iterrows():
                    st.markdown(
    f"**{row['name']}** by {', '.join(row['artists'])}  \nSimilarity: {row['similarity']:.2f}"
)

