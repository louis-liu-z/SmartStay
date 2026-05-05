# SmartStay — California Airbnb Recommender

A Streamlit web app that recommends California Airbnb listings using a hybrid recommendation system combining content-based filtering, collaborative filtering, popularity scoring, and recency signals.

## Features

- **Hybrid recommendations** — blends four signals (content, collaborative, popularity, recency) with adjustable weights
- **Interactive filters** — city, budget, room type, property type, guest count, amenities, and minimum rating
- **AI explanations** — Gemini AI summarizes why each listing matches your preferences
- **Data explorer** — charts for price distribution, room types, listings by city, and ratings
- **Model comparison** — evaluates all five recommendation strategies side-by-side using Precision@K, Coverage, Diversity, and Novelty

## Recommendation Models

| Model | Description |
|---|---|
| Baseline (Popularity) | Ranks by Bayesian-smoothed review score |
| Content-Based | Cosine similarity between user preferences and listing features |
| Collaborative Filtering | Item-item similarity via k-nearest neighbors |
| Hybrid (Static) | Weighted blend of content + collaborative + popularity |
| Hybrid (Time-Based) | Hybrid + recency signal from reviews per month |

## Project Structure

```
SmartStay/
├── app.py                  # Streamlit entry point
├── recommender.py          # Re-exports from src/ for compatibility
├── requirements.txt
└── src/
    ├── constants.py        # Cities, room types, amenities lists
    ├── recommender.py      # SmartStayRecommender class
    ├── gemini.py           # Gemini AI explanation helper
    └── views/
        ├── sidebar.py      # Sidebar UI
        ├── tab_recommendations.py
        ├── tab_explore.py
        └── tab_compare.py
```

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/louis-liu-z/SmartStay.git
   cd SmartStay
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add the dataset**

   Download `clean_ca_df.csv` and place it in the project root. The file is not included in the repo due to its size (167 MB).

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Usage

1. Set your preferences in the sidebar (city, budget, room type, amenities, etc.)
2. Click **Find Stays** to get recommendations
3. Optionally enter a Google API key under *AI Explanations* to generate Gemini summaries
4. Switch to **Compare Models** to benchmark all five recommendation strategies against your filters

## Dataset

California Airbnb listings sourced from [Inside Airbnb](http://insideairbnb.com). Covers 8 cities: Los Angeles, Oakland, Pacific Grove, San Diego, San Francisco, San Mateo County, Santa Clara County, and Santa Cruz County.

## Tech Stack

- [Streamlit](https://streamlit.io) — web app framework
- [scikit-learn](https://scikit-learn.org) — TF-IDF, cosine similarity, k-NN
- [Plotly](https://plotly.com) — interactive charts
- [Google Gemini](https://ai.google.dev) — AI-generated explanations (optional)
