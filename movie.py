"""
Movie Recommendation System using scikit-learn + TMDB API
==========================================================
Approach: Content-Based Filtering using TF-IDF + Cosine Similarity
Data Source: The Movie Database (TMDB) API v3

Requirements:
    pip install pandas numpy scikit-learn requests

Usage:
    1. Set your TMDB API key as an environment variable:
           set TMDB_API_KEY=your_api_key_here          (Windows)
           export TMDB_API_KEY=your_api_key_here       (Linux/Mac)
    2. Run:
           python movie.py
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ── Fix Windows console encoding for emoji/unicode ──
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────

TMDB_BASE_URL = "https://api.themoviedb.org/3"
# Number of pages to fetch from TMDB (20 movies per page)
# Adjust this to get more/fewer movies (max useful: ~500 pages)
PAGES_TO_FETCH = 25  # → ~500 movies
CACHE_FILE = "tmdb_movie_cache.json"  # local cache to avoid re-fetching


# ─────────────────────────────────────────────────────────
# 1. TMDB API — DATA FETCHING
# ─────────────────────────────────────────────────────────

def get_api_key():
    """Retrieve the TMDB API key from environment variables."""
    return os.environ.get("TMDB_API_KEY", "").strip()


def _tmdb_get(endpoint, api_key, params=None):
    """
    Make a GET request to the TMDB API with rate-limit handling.
    """
    url = f"{TMDB_BASE_URL}/{endpoint}"
    if params is None:
        params = {}
    params["api_key"] = api_key

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 401:
                print(f"   ❌  API Unauthorized (401) for {endpoint}. Check API key.")
                return None
            elif resp.status_code == 429:
                # Rate-limited — wait and retry
                wait = int(resp.headers.get("Retry-After", 2))
                print(f"   ⏳ Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"   ⚠️  API returned {resp.status_code} for {endpoint}")
                return None
        except requests.RequestException as e:
            print(f"   ⚠️  Request error: {e}")
            time.sleep(1)
    return None


def fetch_genre_map(api_key):
    """Fetch the genre ID → name mapping from TMDB."""
    data = _tmdb_get("genre/movie/list", api_key, {"language": "en-US"})
    if data and "genres" in data:
        return {g["id"]: g["name"] for g in data["genres"]}
    return {}


def fetch_movie_credits(movie_id, api_key):
    """Fetch cast and crew (director) for a single movie."""
    data = _tmdb_get(f"movie/{movie_id}/credits", api_key)
    if not data:
        return [], ""

    # Top 5 cast members
    cast = [member["name"] for member in (data.get("cast") or [])[:5]]

    # Director from crew
    director = ""
    for member in (data.get("crew") or []):
        if member.get("job") == "Director":
            director = member["name"]
            break

    return cast, director


def fetch_movie_keywords(movie_id, api_key):
    """Fetch keywords for a single movie."""
    data = _tmdb_get(f"movie/{movie_id}/keywords", api_key)
    if data and "keywords" in data:
        return [kw["name"] for kw in data["keywords"]]
    return []


def fetch_movies_from_tmdb(api_key, pages=PAGES_TO_FETCH):
    """
    Fetch popular movies from TMDB, including credits and keywords.
    Returns a list of movie dictionaries.
    """
    genre_map = fetch_genre_map(api_key)
    if not genre_map:
        return None
    print(f"   🎭 Loaded {len(genre_map)} genres from TMDB")

    movies = []
    seen_ids = set()

    print(f"   📥 Fetching {pages} pages of popular movies...")

    for page in range(1, pages + 1):
        data = _tmdb_get("movie/popular", api_key, {
            "language": "en-US",
            "page": page,
        })
        if not data or "results" not in data:
            print(f"   ⚠️  Failed to fetch page {page}, stopping.")
            break

        for movie in data["results"]:
            mid = movie["id"]
            if mid in seen_ids:
                continue
            seen_ids.add(mid)

            # Resolve genre IDs to names
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
            })

        # Progress
        if page % 5 == 0 or page == pages:
            print(f"   ... page {page}/{pages} — {len(movies)} movies collected")
        time.sleep(0.25)  # be polite to the API

    # Fetch credits + keywords for each movie
    total = len(movies)
    print(f"\n   🎬 Fetching credits & keywords for {total} movies...")

    for i, movie in enumerate(movies):
        cast, director = fetch_movie_credits(movie["id"], api_key)
        keywords = fetch_movie_keywords(movie["id"], api_key)

        movie["cast"] = cast
        movie["director"] = director
        movie["keywords"] = keywords

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"   ... {i + 1}/{total} movies enriched")
        time.sleep(0.15)  # rate-limit courtesy

    return movies


import ast

def _load_fallback_data():
    """Load the offline movie dataset if API key is missing or fails."""
    OFFLINE_CSV = "movie_dataset.csv"
    if not os.path.exists(OFFLINE_CSV):
        print(f"   ❌ Fallback dataset {OFFLINE_CSV} not found.")
        return []
    
    print(f"   📂 Loading local dataset from {OFFLINE_CSV}...")
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
        
    print(f"   ✅ Loaded {len(movies)} movies from local dataset")
    return movies

def load_data(api_key):
    """
    Load movie data — uses local cache if available, otherwise fetches
    fresh data from TMDB. Falls back to local CSV if TMDB fails or no key.
    """
    # Check for cached data
    if os.path.exists(CACHE_FILE):
        print(f"   📦 Loading cached data from {CACHE_FILE}...")
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            movies = json.load(f)
        print(f"   ✅ Loaded {len(movies)} movies from cache")
        return pd.DataFrame(movies)

    movies = []
    # Fetch from TMDB if API key is provided
    if api_key:
        print("   🌐 No cache found — fetching from TMDB API...")
        movies = fetch_movies_from_tmdb(api_key)

    if not movies:
        print("   ⚠️ No TMDB API key or fetching failed! Falling back to local dataset...")
        movies = _load_fallback_data()
        
    if not movies:
        print("   ❌ No movies available! Check your API key or ensure movie_dataset.csv is present.")
        sys.exit(1)

    # Save cache if we fetched from TMDB
    if api_key and not os.path.exists(CACHE_FILE) and movies and movies[0].get("poster_path"):
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(movies, f, ensure_ascii=False, indent=2)
        print(f"   💾 Cached {len(movies)} movies to {CACHE_FILE}")

    return pd.DataFrame(movies)


# ─────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────────────────

def preprocess(df):
    """
    Clean and engineer features for content-based recommendation.
    Creates a text 'soup' combining genres, keywords, cast, director, and overview.
    """
    df = df.copy()

    # ── Convert list columns to space-separated strings ──
    def list_to_str(val):
        if isinstance(val, list):
            return " ".join(str(v) for v in val)
        if isinstance(val, str):
            return val
        return ""

    df["genres_clean"] = df["genres"].apply(list_to_str)
    df["keywords_clean"] = df["keywords"].apply(list_to_str)
    df["cast_clean"] = df["cast"].apply(list_to_str)
    df["director"] = df["director"].fillna("").astype(str)
    df["overview"] = df["overview"].fillna("").astype(str)

    # ── Remove spaces in multi-word names to treat as single tokens ──
    # e.g., "Christopher Nolan" → "christophernolan"
    df["cast_tokens"] = df["cast_clean"].apply(
        lambda x: " ".join(name.replace(" ", "").lower() for name in x.split(",")) if "," in x
        else " ".join(name.replace(" ", "").lower() for name in (x.split("  ") if "  " in x else [x]))
    )
    df["director_token"] = df["director"].apply(lambda x: x.replace(" ", "").lower())

    # ── Build the combined text 'soup' with feature weighting ──
    # Genres × 2 (important signal), Director × 2, keywords, cast, overview
    df["soup"] = (
        df["genres_clean"] + " " + df["genres_clean"] + " " +   # genres × 2
        df["keywords_clean"] + " " +
        df["cast_tokens"] + " " +
        df["director_token"] + " " + df["director_token"] + " " +  # director × 2
        df["overview"]
    )
    df["soup"] = df["soup"].str.lower().str.replace(r"[^a-z0-9\s]", "", regex=True)
    df["soup"] = df["soup"].str.replace(r"\s+", " ", regex=True).str.strip()

    # ── Title normalization ──
    df["title"] = df["title"].fillna("Unknown")

    print(f"   ✅ Preprocessed {len(df)} movies")
    print(f"   Features: genres, keywords, cast (top 5), director, overview")

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────
# 3. MODEL BUILDING (TF-IDF + Cosine Similarity)
# ─────────────────────────────────────────────────────────

def build_model(df):
    """
    Build the recommendation model:
      1. TF-IDF vectorization of the text 'soup'
      2. Cosine similarity matrix between all movies

    Returns:
        cosine_sim   : np.ndarray — pairwise similarity matrix
        title_to_idx : dict       — lowercased title → row index
        tfidf        : fitted TfidfVectorizer
    """
    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),      # unigrams + bigrams for better context
        max_features=15000,      # cap vocabulary size
        sublinear_tf=True,       # apply log normalization to TF
        min_df=2,                # ignore very rare terms
    )

    tfidf_matrix = tfidf.fit_transform(df["soup"])
    print(f"   ✅ TF-IDF matrix: {tfidf_matrix.shape[0]} movies × {tfidf_matrix.shape[1]} features")

    # Pairwise cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print(f"   ✅ Cosine similarity matrix: {cosine_sim.shape}")

    # Title → index mapping (lowercase for case-insensitive lookup)
    title_to_idx = {}
    for idx, title in enumerate(df["title"]):
        key = title.lower()
        if key not in title_to_idx:  # keep first occurrence for duplicates
            title_to_idx[key] = idx

    return cosine_sim, title_to_idx, tfidf


# ─────────────────────────────────────────────────────────
# 4. CONTENT-BASED RECOMMENDATIONS
# ─────────────────────────────────────────────────────────

def get_recommendations(title, df, cosine_sim, title_to_idx, top_n=10):
    """
    Return top-N movies most similar to the given title using
    pure content-based cosine similarity.

    Parameters
    ----------
    title : str
        Movie title to find recommendations for.
    df : pd.DataFrame
        Preprocessed movie dataframe.
    cosine_sim : np.ndarray
        Pairwise cosine similarity matrix.
    title_to_idx : dict
        Mapping from lowercased title → row index.
    top_n : int
        Number of recommendations to return.

    Returns
    -------
    pd.DataFrame with columns: rank, title, genres, director, vote_avg, similarity_score
    """
    key = title.lower().strip()

    # Fuzzy fallback: substring match
    if key not in title_to_idx:
        matches = [t for t in title_to_idx if key in t]
        if not matches:
            # Try word-by-word matching
            words = key.split()
            matches = [t for t in title_to_idx if all(w in t for w in words)]
        if not matches:
            print(f"   ❌ '{title}' not found in dataset.")
            print(f"   Available titles (sample):")
            for t in list(df["title"].head(20)):
                print(f"      • {t}")
            return pd.DataFrame()
        key = matches[0]
        print(f"   🔍 Closest match: '{df.loc[title_to_idx[key], 'title']}'")

    idx = title_to_idx[key]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]

    movie_indices = [s[0] for s in sim_scores]
    scores = [round(s[1], 4) for s in sim_scores]

    result = df.loc[movie_indices, ["title", "genres_clean", "director", "vote_average"]].copy()
    result.insert(0, "rank", range(1, len(result) + 1))
    result["similarity_score"] = scores
    result.columns = ["rank", "title", "genres", "director", "vote_avg", "similarity_score"]
    return result.reset_index(drop=True)


# ─────────────────────────────────────────────────────────
# 5. HYBRID RECOMMENDATIONS (Content + Popularity + Rating)
# ─────────────────────────────────────────────────────────

def hybrid_recommendations(title, df, cosine_sim, title_to_idx, top_n=10):
    """
    Combine content similarity with popularity & rating for a hybrid score.

    hybrid_score = 0.60 × cosine_sim
                 + 0.25 × normalized_vote_average
                 + 0.15 × normalized_popularity
    """
    key = title.lower().strip()
    if key not in title_to_idx:
        matches = [t for t in title_to_idx if key in t]
        if not matches:
            print(f"   ❌ '{title}' not found.")
            return pd.DataFrame()
        key = matches[0]
        print(f"   🔍 Closest match: '{df.loc[title_to_idx[key], 'title']}'")

    idx = title_to_idx[key]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:50]  # candidate pool

    movie_indices = [s[0] for s in sim_scores]
    scores = [s[1] for s in sim_scores]

    subset = df.loc[movie_indices].copy()
    subset["content_sim"] = scores

    scaler = MinMaxScaler()

    # Normalize vote_average and popularity to [0, 1]
    if "vote_average" in subset.columns and subset["vote_average"].nunique() > 1:
        subset["norm_vote"] = scaler.fit_transform(subset[["vote_average"]])
    else:
        subset["norm_vote"] = 0.5

    if "popularity" in subset.columns and subset["popularity"].nunique() > 1:
        subset["norm_pop"] = scaler.fit_transform(subset[["popularity"]])
    else:
        subset["norm_pop"] = 0.5

    # Weighted hybrid score
    subset["hybrid_score"] = (
        0.60 * subset["content_sim"] +
        0.25 * subset["norm_vote"] +
        0.15 * subset["norm_pop"]
    )

    subset = subset.sort_values("hybrid_score", ascending=False).head(top_n)
    subset.insert(0, "rank", range(1, len(subset) + 1))

    cols = ["rank", "title", "genres_clean", "director", "vote_average",
            "content_sim", "hybrid_score"]
    cols = [c for c in cols if c in subset.columns]
    result = subset[cols].reset_index(drop=True)
    result.columns = (
        ["rank", "title", "genres", "director", "vote_avg",
         "content_sim", "hybrid_score"][:len(result.columns)]
    )
    return result


# ─────────────────────────────────────────────────────────
# 6. MODEL EVALUATION (Coverage + Diversity)
# ─────────────────────────────────────────────────────────

def evaluate_model(df, cosine_sim, title_to_idx, sample_size=10):
    """
    Evaluate the recommendation model using:
      - Catalog Coverage  : % of unique movies appearing in any rec list
      - Avg. Intra-List Diversity : avg pairwise dissimilarity within rec lists
    """
    all_recs = set()
    diversities = []

    sample_titles = df["title"].sample(
        min(sample_size, len(df)), random_state=42
    ).tolist()

    for t in sample_titles:
        recs = get_recommendations(t, df, cosine_sim, title_to_idx, top_n=10)
        if recs.empty:
            continue
        rec_indices = [
            title_to_idx[r.lower()]
            for r in recs["title"]
            if r.lower() in title_to_idx
        ]
        all_recs.update(rec_indices)

        # Intra-list diversity: average pairwise distance (1 - similarity)
        if len(rec_indices) > 1:
            sub = cosine_sim[np.ix_(rec_indices, rec_indices)]
            n = len(rec_indices)
            diversity = 1 - (sub.sum() - n) / (n * (n - 1))
            diversities.append(diversity)

    coverage = len(all_recs) / len(df) * 100
    avg_diversity = np.mean(diversities) if diversities else 0

    print(f"   Catalog Coverage  : {coverage:.1f}%")
    print(f"   Avg. Diversity    : {avg_diversity:.4f}  (1 = max diverse, 0 = identical)")
    return {"coverage": coverage, "avg_diversity": avg_diversity}


# ─────────────────────────────────────────────────────────
# 7. INTERACTIVE MODE
# ─────────────────────────────────────────────────────────

def interactive_mode(df, cosine_sim, title_to_idx):
    """Let the user search for recommendations interactively."""
    print("\n" + "=" * 65)
    print("  🎬  INTERACTIVE MODE — type a movie title to get recommendations")
    print("  Type 'quit' to exit, 'list' to see all titles")
    print("=" * 65)

    while True:
        print()
        user_input = input("  🎬 Enter movie title: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("  👋 Goodbye!")
            break
        if user_input.lower() == "list":
            print("\n  Available movies:")
            for i, t in enumerate(sorted(df["title"].unique()), 1):
                print(f"    {i:>4}. {t}")
            continue

        print(f"\n  {'─' * 60}")
        print(f"  📽  Content-Based Recommendations for: '{user_input}'")
        print(f"  {'─' * 60}")
        recs = get_recommendations(user_input, df, cosine_sim, title_to_idx, top_n=10)
        if not recs.empty:
            print(recs.to_string(index=False))

        print(f"\n  {'─' * 60}")
        print(f"  🔀  Hybrid Recommendations for: '{user_input}'")
        print(f"  {'─' * 60}")
        hybrid = hybrid_recommendations(user_input, df, cosine_sim, title_to_idx, top_n=10)
        if not hybrid.empty:
            print(hybrid.to_string(index=False))


# ─────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_colwidth", 35)

    print("=" * 65)
    print("  🎬  Movie Recommendation System")
    print("  scikit-learn TF-IDF │ TMDB API")
    print("=" * 65)
    print()

    # ── 1. Get API key ──
    api_key = get_api_key()
    if api_key:
        print(f"   🔑 TMDB API key loaded\n")
    else:
        print(f"   ⚠️ No TMDB API key found, will use offline dataset\n")

    # ── 2. Load data from TMDB (or cache) ──
    print("── STEP 1: Loading Data ──")
    df_raw = load_data(api_key)
    print()

    # ── 3. Preprocess ──
    print("── STEP 2: Preprocessing ──")
    df = preprocess(df_raw)
    print()

    # ── 4. Build model ──
    print("── STEP 3: Building Model ──")
    cosine_sim, title_to_idx, tfidf = build_model(df)
    print()

    # ── 5. Demo recommendations ──
    demo_titles = ["Inception", "The Avengers", "Interstellar"]
    for demo_title in demo_titles:
        if demo_title.lower() in title_to_idx:
            print(f"{'─' * 65}")
            print(f"  📽  Content-Based Recommendations for: '{demo_title}'")
            print(f"{'─' * 65}")
            recs = get_recommendations(demo_title, df, cosine_sim, title_to_idx, top_n=5)
            if not recs.empty:
                print(recs.to_string(index=False))
            print()

            print(f"{'─' * 65}")
            print(f"  🔀  Hybrid Recommendations for: '{demo_title}'")
            print(f"{'─' * 65}")
            hybrid = hybrid_recommendations(demo_title, df, cosine_sim, title_to_idx, top_n=5)
            if not hybrid.empty:
                print(hybrid.to_string(index=False))
            print()
            break  # just one demo by default

    # ── 6. Evaluate ──
    print("── STEP 4: Model Evaluation ──")
    evaluate_model(df, cosine_sim, title_to_idx, sample_size=10)
    print()

    # ── 7. Interactive mode ──
    interactive_mode(df, cosine_sim, title_to_idx)