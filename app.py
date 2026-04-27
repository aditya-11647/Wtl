"""
Movie Recommendation System — Flask Web Application
=====================================================
Uses TMDB API for data, scikit-learn TF-IDF for recommendations.
"""

import os
import sys
import time
import json
import threading
import pandas as pd
import numpy as np
import requests
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
load_dotenv()

# ── Fix Windows console encoding ──
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

app = Flask(__name__, 
            static_folder=os.path.abspath("frontend/dist"), 
            static_url_path="/", 
            template_folder=os.path.abspath("frontend/dist"))
CORS(app)

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────

TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
print("API KEY:", TMDB_API_KEY)

if not TMDB_API_KEY:
    raise ValueError("TMDB API key not set")

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"

PAGES_TO_FETCH = 27       

CACHE_FILE = "tmdb_movie_cache.json"
OFFLINE_CSV = "movie_dataset.csv"

# Global state for the recommendation engine
engine = {
    "df": None,
    "cosine_sim": None,
    "title_to_idx": None,
    "ready": False,
    "status": "Initializing...",
    "progress": 0,
}


# ─────────────────────────────────────────────────────────
# TMDB API HELPERS
# ─────────────────────────────────────────────────────────

def _tmdb_get(endpoint, params=None):
    print(f"[TMDB] Calling: {endpoint}")
    

    url = f"{TMDB_BASE_URL}/{endpoint}"
    if params is None:
        params = {}
    params["api_key"] = TMDB_API_KEY
    for attempt in range(5):
        try:
            engine["status"] = f"Fetching {endpoint}... (Attempt {attempt+1}/5)"
            resp = requests.get(url, params=params, timeout=30)
            print("STATUS CODE:", resp.status_code)
            print("RESPONSE:", resp.text[:200])
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 401:
                engine["status"] = "Invalid TMDB API key."
                print(f"[ENGINE] API Unauthorized (401) on {endpoint}. Aborting fetch.")
                return None
            elif resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 5))
                engine["status"] = f"Rate limited. Waiting {wait}s..."
                time.sleep(wait)
            else:
                time.sleep(2)
        except requests.RequestException as e:
            print("ERROR:", e)   # 👈 ADD THIS
            engine["status"] = f"Network error: {e}"
            time.sleep(2)
    return None
    


def fetch_genre_map():
    data = _tmdb_get("genre/movie/list", {"language": "en-US"})
    if data and "genres" in data:
        return {g["id"]: g["name"] for g in data["genres"]}
    return {}


def fetch_movie_credits(movie_id):
    data = _tmdb_get(f"movie/{movie_id}/credits")
    if not data:
        return [], ""
    cast = [m["name"] for m in (data.get("cast") or [])[:5]]
    director = ""
    for m in (data.get("crew") or []):
        if m.get("job") == "Director":
            director = m["name"]
            break
    return cast, director


def fetch_movie_keywords(movie_id):
    data = _tmdb_get(f"movie/{movie_id}/keywords")
    if data and "keywords" in data:
        return [kw["name"] for kw in data["keywords"]]
    return []


def fetch_movies_from_tmdb(pages=PAGES_TO_FETCH):
    genre_map = fetch_genre_map()
    if not genre_map:
        return None
    engine["status"] = f"Loaded {len(genre_map)} genres"

    movies = []
    seen_ids = set()

    for page in range(1, pages + 1):
        data = _tmdb_get("movie/now_playing", {"language": "en-US", "page": page})
        if not data:
            data = _tmdb_get("movie/popular", {"language": "en-US", "page": page})
        if not data or "results" not in data:
            continue
        for movie in data["results"]:
            mid = movie["id"]
            if mid in seen_ids:
                continue
            seen_ids.add(mid)
            genres = [genre_map.get(gid, "") for gid in movie.get("genre_ids", [])]
            genres = [g for g in genres if g]
            movies.append({
                "id": mid,
                "title": movie.get("title", "Unknown"),
                "overview": movie.get("overview", ""),
                "genres": genres,
                "popularity": movie.get("popularity", 0),
                "vote_average": movie.get("vote_average", 0),
                "vote_count": movie.get("vote_count", 0),
                "release_date": movie.get("release_date", ""),
                "poster_path": movie.get("poster_path", ""),
                "cast": [],
                "director": "",
                "keywords": [] 
            })
        engine["progress"] = int((page / pages) * 30)
        engine["status"] = f"Fetching movies... page {page}/{pages}"
        

    total = len(movies)
    
        

    return movies


# ─────────────────────────────────────────────────────────
# PREPROCESSING & MODEL
# ─────────────────────────────────────────────────────────

def list_to_str(val):
    if isinstance(val, list):
        return " ".join(str(v) for v in val)
    if isinstance(val, str):
        return val
    return ""


def preprocess(df):
    df = df.copy()
    df["genres_clean"] = df["genres"].apply(list_to_str)
    df["keywords_clean"] = df["keywords"].apply(list_to_str)
    df["cast_clean"] = df["cast"].apply(list_to_str)
    df["director"] = df["director"].fillna("").astype(str)
    df["overview"] = df["overview"].fillna("").astype(str)

    df["cast_tokens"] = df["cast_clean"].apply(
        lambda x: " ".join(n.replace(" ", "").lower() for n in (x.split(",") if "," in x else [x]))
    )
    df["director_token"] = df["director"].apply(lambda x: x.replace(" ", "").lower())

    df["soup"] = (
        df["genres_clean"] + " " + df["genres_clean"] + " " +
        df["keywords_clean"] + " " +
        df["cast_tokens"] + " " +
        df["director_token"] + " " + df["director_token"] + " " +
        df["overview"]
    )
    df["soup"] = df["soup"].str.lower().str.replace(r"[^a-z0-9\s]", "", regex=True)
    df["soup"] = df["soup"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["title"] = df["title"].fillna("Unknown")
    return df.reset_index(drop=True)


def build_model(df):
    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=15000,
        sublinear_tf=True,
        min_df=2,
    )
    tfidf_matrix = tfidf.fit_transform(df["soup"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    title_to_idx = {}
    for idx, title in enumerate(df["title"]):
        key = title.lower()
        if key not in title_to_idx:
            title_to_idx[key] = idx
    return cosine_sim, title_to_idx


# ─────────────────────────────────────────────────────────
# RECOMMENDATION FUNCTIONS
# ─────────────────────────────────────────────────────────

def find_movie_index(title, title_to_idx):
    key = title.lower().strip()
    if key in title_to_idx:
        return title_to_idx[key]
    matches = [t for t in title_to_idx if key in t]
    if not matches:
        words = key.split()
        matches = [t for t in title_to_idx if all(w in t for w in words)]
    if matches:
        return title_to_idx[matches[0]]
    return None


def get_recommendations(title, df, cosine_sim, title_to_idx, top_n=12):
    idx = find_movie_index(title, title_to_idx)
    if idx is None:
        return []

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]

    results = []
    for rank, (movie_idx, score) in enumerate(sim_scores, 1):
        row = df.iloc[movie_idx]
        poster = f"{TMDB_IMG_BASE}{row['poster_path']}" if row.get("poster_path") else ""
        results.append({
            "rank": rank,
            "id": int(row.get("id", 0)),
            "title": row["title"],
            "genres": row.get("genres_clean", ""),
            "director": row.get("director", ""),
            "vote_average": round(float(row.get("vote_average", 0)), 1),
            "release_date": row.get("release_date", ""),
            "overview": row.get("overview", "")[:200],
            "poster": poster,
            "similarity": round(float(score), 4),
        })
    return results


def hybrid_recommendations(title, df, cosine_sim, title_to_idx, top_n=12):
    idx = find_movie_index(title, title_to_idx)
    if idx is None:
        return []

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:50]

    movie_indices = [s[0] for s in sim_scores]
    scores = [s[1] for s in sim_scores]

    subset = df.loc[movie_indices].copy()
    subset["content_sim"] = scores

    scaler = MinMaxScaler()
    if subset["vote_average"].nunique() > 1:
        subset["norm_vote"] = scaler.fit_transform(subset[["vote_average"]])
    else:
        subset["norm_vote"] = 0.5
    if subset["popularity"].nunique() > 1:
        subset["norm_pop"] = scaler.fit_transform(subset[["popularity"]])
    else:
        subset["norm_pop"] = 0.5

    subset["hybrid_score"] = (
        0.60 * subset["content_sim"] +
        0.25 * subset["norm_vote"] +
        0.15 * subset["norm_pop"]
    )
    subset = subset.sort_values("hybrid_score", ascending=False).head(top_n)

    results = []
    for rank, (_, row) in enumerate(subset.iterrows(), 1):
        poster = f"{TMDB_IMG_BASE}{row['poster_path']}" if row.get("poster_path") else "https://via.placeholder.com/300x450?text=No+Image"
        results.append({
            "rank": rank,
            "id": int(row.get("id", 0)),
            "title": row["title"],
            "genres": row.get("genres_clean", ""),
            "director": row.get("director", ""),
            "vote_average": round(float(row.get("vote_average", 0)), 1),
            "release_date": row.get("release_date", ""),
            "overview": row.get("overview", "")[:200],
            "poster": poster,
            "similarity": round(float(row.get("content_sim", 0)), 4),
            "hybrid_score": round(float(row.get("hybrid_score", 0)), 4),
        })
    return results


# ─────────────────────────────────────────────────────────
# BACKGROUND INITIALIZATION
# ─────────────────────────────────────────────────────────

import ast

def safe_parse(val):
    try:
        return ast.literal_eval(val)
    except:
        return val

def _load_fallback_data():
    """Load the full 4800 movie dataset if API fails."""
    if not os.path.exists(OFFLINE_CSV):
        return []
    
    df = pd.read_csv(OFFLINE_CSV)
    
    movies = []
    for _, row in df.iterrows():
        # The CSV has stringified JSON lists for genres and keywords
        genres = []
        try:
            g_data = ast.literal_eval(row.get("genres", "[]"))
            if isinstance(g_data, list):
                genres = [g.get("name", "") for g in g_data if isinstance(g, dict)]
            elif isinstance(g_data, str):
                genres = g_data.split()
        except:
            genres = str(row.get("genres", "")).split()
            
        keywords = []
        try:
            k_data = ast.literal_eval(row.get("keywords", "[]"))
            if isinstance(k_data, list):
                keywords = [k.get("name", "") for k in k_data if isinstance(k, dict)]
            elif isinstance(k_data, str):
                keywords = k_data.split()
        except:
            keywords = str(row.get("keywords", "")).split()
            
        cast = []
        try:
            c_data = ast.literal_eval(row.get("cast", "[]"))
            if isinstance(c_data, list):
                cast = [c.get("name", "") for c in c_data if isinstance(c, dict)][:5]
            elif isinstance(c_data, str):
                cast = c_data.split()[:5]
        except:
            cast = str(row.get("cast", "")).split()[:5]
            
        director = ""
        try:
            cr_data = ast.literal_eval(row.get("crew", "[]"))
            if isinstance(cr_data, list):
                for c in cr_data:
                    if isinstance(c, dict) and c.get("job") == "Director":
                        director = c.get("name", "")
                        break
        except:
            pass

        title = row.get("title") or row.get("original_title") or "Unknown"

        movies.append({
            "id": row.get("id", 0),
            "title": title,
            "overview": row.get("overview", ""),
            "genres": genres,
            "keywords": keywords,
            "cast": cast,
            "director": director,
            "popularity": row.get("popularity", 0),
            "vote_average": row.get("vote_average", 0),
            "release_date": str(row.get("release_date", "")),
            "poster_path": ""  # No posters in the CSV
        })
        
    return movies


def initialize_engine():
    """Load data, preprocess, and build model in background thread."""
    try:
        engine["status"] = "Connecting to TMDB API..."
        engine["progress"] = 10
        movies = fetch_movies_from_tmdb()

        if not movies:
            engine["status"] = "TMDB API failed. Check API key or internet."
            print("[ENGINE] TMDB FAILED ❌")
            return
        
        if not movies:
            engine["status"] = "Error: Local dataset not found!"
            print("[ENGINE] Failed to load local dataset.")
            return

        engine["status"] = "Preprocessing..."
        engine["progress"] = 90
        df = pd.DataFrame(movies)
        df = preprocess(df)

        engine["status"] = "Building model..."
        engine["progress"] = 95
        cosine_sim, title_to_idx = build_model(df)

        engine["df"] = df
        engine["cosine_sim"] = cosine_sim
        engine["title_to_idx"] = title_to_idx
        engine["ready"] = True
        engine["progress"] = 100
        engine["status"] = f"Ready — {len(df)} movies loaded"
        print(f"[ENGINE] Ready with {len(df)} movies")

    except Exception as e:
        engine["status"] = f"Error: {e}"
        print(f"[ENGINE] Error: {e}")


# ─────────────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "ready": engine["ready"],
        "status": engine["status"],
        "progress": engine["progress"],
        "movie_count": len(engine["df"]) if engine["df"] is not None else 0,
    })


@app.route("/api/search")
def api_search():
    if not engine["ready"]:
        return jsonify({"error": "Engine not ready"}), 503

    query = request.args.get("q", "").strip().lower()
    if not query or len(query) < 2:
        return jsonify({"results": []})

    df = engine["df"]
    matches = df[df["title"].str.lower().str.contains(query, regex=False, na=False)].head(10)

    results = []
    for _, row in matches.iterrows():
        poster = f"{TMDB_IMG_BASE}{row['poster_path']}" if row.get("poster_path") else "https://via.placeholder.com/300x450?text=No+Image"
        results.append({
            "id": int(row.get("id", 0)),
            "title": row["title"],
            "genres": row.get("genres_clean", ""),
            "vote_average": round(float(row.get("vote_average", 0)), 1),
            "release_date": row.get("release_date", ""),
            "poster": poster,
        })
    return jsonify({"results": results})


@app.route("/api/recommend")
def api_recommend():
    if not engine["ready"]:
        return jsonify({"error": "Engine not ready"}), 503

    title = request.args.get("title", "").strip()
    mode = request.args.get("mode", "content")  # "content" or "hybrid"
    top_n = min(int(request.args.get("n", 12)), 20)

    if not title:
        return jsonify({"error": "No title provided"}), 400

    df = engine["df"]
    cosine_sim = engine["cosine_sim"]
    title_to_idx = engine["title_to_idx"]

    if mode == "hybrid":
        recs = hybrid_recommendations(title, df, cosine_sim, title_to_idx, top_n)
    else:
        recs = get_recommendations(title, df, cosine_sim, title_to_idx, top_n)

    # Get the source movie info
    idx = find_movie_index(title, title_to_idx)
    source = None
    if idx is not None:
        row = df.iloc[idx]
        poster = f"{TMDB_IMG_BASE}{row['poster_path']}" if row.get("poster_path") else ""
        source = {
            "title": row["title"],
            "genres": row.get("genres_clean", ""),
            "director": row.get("director", ""),
            "vote_average": round(float(row.get("vote_average", 0)), 1),
            "overview": row.get("overview", ""),
            "poster": poster,
        }

    return jsonify({
        "source": source,
        "recommendations": recs,
        "mode": mode,
        "count": len(recs),
    })


@app.route("/api/trending")
def api_trending():
    """Return top-rated movies from the dataset for the homepage."""
    if not engine["ready"]:
        return jsonify({"error": "Engine not ready"}), 503

    df = engine["df"]
    trending = df.nlargest(12, "popularity")

    results = []
    for _, row in trending.iterrows():
        poster = f"{TMDB_IMG_BASE}{row['poster_path']}" if row.get("poster_path") else ""
        results.append({
            "id": int(row.get("id", 0)),
            "title": row["title"],
            "genres": row.get("genres_clean", ""),
            "vote_average": round(float(row.get("vote_average", 0)), 1),
            "release_date": row.get("release_date", ""),
            "poster": poster,
        })
    return jsonify({"results": results})

@app.route("/api/live-search")
def live_search():
    query = request.args.get("q")

    data = _tmdb_get("search/movie", {
        "language": "en-US",
        "query": query
    })

    results = []

    if data and "results" in data:
        for movie in data["results"][:10]:
            results.append({
                "title": movie.get("title"),
                "poster": f"{TMDB_IMG_BASE}{movie.get('poster_path')}" if movie.get("poster_path") else "",
                "release_date": movie.get("release_date"),
                "rating": movie.get("vote_average")
            })

    return jsonify({"results": results})
# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Movie Recommendation System — Web App")
    print("  scikit-learn TF-IDF | TMDB API | Flask")
    print("=" * 55)

    # Start engine initialization in background
    init_thread = threading.Thread(target=initialize_engine, daemon=True)
    init_thread.start()

    app.run(debug=False, host="127.0.0.1", port=5000)
