import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack, csr_matrix


class SmartStayRecommender:
    def __init__(self, data_path="clean_ca_df.csv"):
        self._build(data_path)

    def _build(self, data_path):
        df = pd.read_csv(data_path, engine="python")
        self.raw = df.copy()

        num_cols = ["price", "beds", "bathrooms", "accommodates", "review_scores_rating"]
        for col in num_cols:
            self.raw[col] = pd.to_numeric(self.raw[col], errors="coerce")
        self.num_cols = num_cols

        self.scaler = MinMaxScaler()
        num_matrix = self.scaler.fit_transform(self.raw[num_cols].fillna(0))

        cat_df = pd.get_dummies(
            self.raw[["room_type", "property_type"]].fillna("Unknown"), drop_first=False
        )
        self.cat_cols = cat_df.columns.tolist()
        cat_matrix = cat_df.values

        self.tfidf = TfidfVectorizer(max_features=100, stop_words="english")
        amenity_matrix = self.tfidf.fit_transform(self.raw["amenities_clean"].fillna(""))

        self.content_matrix = hstack(
            [csr_matrix(num_matrix), csr_matrix(cat_matrix), amenity_matrix]
        )

        C = self.raw["review_scores_rating"].fillna(
            self.raw["review_scores_rating"].mean()
        ).mean()
        m = self.raw["number_of_reviews"].fillna(0).quantile(0.25)

        def _bayesian(row):
            v = row["number_of_reviews"] if not pd.isna(row["number_of_reviews"]) else 0
            R = row["review_scores_rating"] if not pd.isna(row["review_scores_rating"]) else C
            return (v / (v + m + 1e-9)) * R + (m / (v + m + 1e-9)) * C

        self.raw["popularity_score"] = self.raw.apply(_bayesian, axis=1)

        rpm = self.raw["reviews_per_month"].fillna(0).values
        self.raw["recency_weight"] = rpm / (rpm.max() + 1e-9)
        self.raw["time_aware_score"] = (
            self.raw["popularity_score"] * (1 + self.raw["recency_weight"])
        )

        self.item_nn = NearestNeighbors(metric="cosine", algorithm="brute")
        self.item_nn.fit(self.content_matrix)

    @staticmethod
    def _norm(values):
        v = np.asarray(values, dtype=float)
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo + 1e-9)

    def build_user_vector(
        self, price, beds, bathrooms, accommodates,
        min_rating, room_type, property_type, amenities_str
    ):
        user_num = self.scaler.transform(
            pd.DataFrame(
                [[price, beds, bathrooms, accommodates, min_rating]],
                columns=self.num_cols,
            ).fillna(0)
        )
        user_cat = np.zeros((1, len(self.cat_cols)))
        for col in [f"room_type_{room_type}", f"property_type_{property_type}"]:
            if col in self.cat_cols:
                user_cat[0, self.cat_cols.index(col)] = 1
        user_amenity = self.tfidf.transform([amenities_str or ""])
        return hstack([csr_matrix(user_num), csr_matrix(user_cat), user_amenity])

    def recommend_popular(self, budget_max=None, city=None, room_type=None, top_n=10):
        df = self.raw.copy()
        if budget_max is not None:
            df = df[df["price"] <= budget_max]
        if city:
            df = df[df["City"].fillna("").str.lower() == city.lower()]
        if room_type:
            df = df[df["room_type"] == room_type]
        cols = ["id", "name", "City", "room_type", "price",
                "review_scores_rating", "number_of_reviews", "popularity_score"]
        return df.nlargest(top_n, "popularity_score")[cols]

    def recommend_content(self, user_vector, budget_max=None, city=None,
                          room_type=None, top_n=10):
        df = self.raw.copy()
        df["content_score"] = cosine_similarity(user_vector, self.content_matrix).flatten()
        if budget_max is not None:
            df = df[df["price"] <= budget_max]
        if city:
            df = df[df["City"].fillna("").str.lower() == city.lower()]
        if room_type:
            df = df[df["room_type"] == room_type]
        cols = ["id", "name", "City", "room_type", "price",
                "review_scores_rating", "content_score"]
        return df.nlargest(top_n, "content_score")[cols]

    def recommend_similar_listings(self, listing_id, top_n=10):
        match = self.raw.index[self.raw["id"] == listing_id]
        if len(match) == 0:
            raise ValueError(f"Listing id {listing_id} not found.")
        idx = match[0]
        distances, indices = self.item_nn.kneighbors(
            self.content_matrix[idx], n_neighbors=top_n + 1
        )
        neighbor_idx = indices.flatten()[1:]
        neighbor_dist = distances.flatten()[1:]
        df = self.raw.iloc[neighbor_idx][
            ["id", "name", "City", "room_type", "price", "review_scores_rating"]
        ].copy()
        df["collab_score"] = 1 - neighbor_dist
        return df.sort_values("collab_score", ascending=False)

    def hybrid_recommend(
        self, user_vector=None, listing_id=None, budget_max=None,
        city=None, room_type=None,
        w_content=0.5, w_collab=0.3, w_popular=0.2, top_n=10,
    ):
        df = self.raw.copy()
        n = len(df)

        content_scores = (
            cosine_similarity(user_vector, self.content_matrix).flatten()
            if user_vector is not None else np.zeros(n)
        )
        collab_scores = np.zeros(n)
        if listing_id is not None:
            match = df.index[df["id"] == listing_id]
            if len(match) > 0:
                collab_scores = cosine_similarity(
                    self.content_matrix[match[0]], self.content_matrix
                ).flatten()

        pop_scores = self._norm(df["popularity_score"].fillna(0).values)

        df["content_score"] = content_scores
        df["collab_score"] = collab_scores
        df["pop_score_norm"] = pop_scores
        df["hybrid_score"] = (
            w_content * content_scores
            + w_collab * collab_scores
            + w_popular * pop_scores
        )

        if budget_max is not None:
            df = df[df["price"] <= budget_max]
        if city:
            df = df[df["City"].fillna("").str.lower() == city.lower()]
        if room_type:
            df = df[df["room_type"] == room_type]

        cols = ["id", "name", "City", "room_type", "price", "review_scores_rating",
                "content_score", "collab_score", "pop_score_norm", "hybrid_score"]
        return df.nlargest(top_n, "hybrid_score")[cols]

    def hybrid_time_recommend(
        self, user_vector=None, listing_id=None, budget_max=None,
        city=None, room_type=None,
        w_content=0.4, w_collab=0.25, w_popular=0.2, w_time=0.15, top_n=10,
    ):
        df = self.raw.copy()
        n = len(df)

        content_scores = (
            cosine_similarity(user_vector, self.content_matrix).flatten()
            if user_vector is not None else np.zeros(n)
        )
        collab_scores = np.zeros(n)
        if listing_id is not None:
            match = df.index[df["id"] == listing_id]
            if len(match) > 0:
                collab_scores = cosine_similarity(
                    self.content_matrix[match[0]], self.content_matrix
                ).flatten()

        pop_scores = self._norm(df["popularity_score"].fillna(0).values)
        time_scores = self._norm(df["time_aware_score"].fillna(0).values)

        df["hybrid_time_score"] = (
            w_content * content_scores
            + w_collab * collab_scores
            + w_popular * pop_scores
            + w_time * time_scores
        )

        if budget_max is not None:
            df = df[df["price"] <= budget_max]
        if city:
            df = df[df["City"].fillna("").str.lower() == city.lower()]
        if room_type:
            df = df[df["room_type"] == room_type]

        cols = ["id", "name", "City", "room_type", "price", "review_scores_rating",
                "amenities_clean", "hybrid_time_score"]
        return df.nlargest(top_n, "hybrid_time_score")[cols]

    def precision_at_k(self, recommended_ids, k=10, threshold=4.5):
        recs = self.raw[self.raw["id"].isin(recommended_ids[:k])]
        return (recs["review_scores_rating"] >= threshold).sum() / k if k > 0 else 0.0

    def coverage(self, all_id_lists):
        unique = set()
        for ids in all_id_lists:
            unique.update(ids)
        return len(unique) / len(self.raw)

    def diversity_at_k(self, recommended_ids, k=10):
        subset = self.raw[self.raw["id"].isin(recommended_ids[:k])]
        if len(subset) < 2:
            return 0.0
        idx = subset.index.tolist()
        sim = cosine_similarity(self.content_matrix[idx])
        np.fill_diagonal(sim, 0)
        return 1 - sim.sum() / (len(idx) * (len(idx) - 1))

    def novelty_at_k(self, recommended_ids, k=10):
        recs = self.raw[self.raw["id"].isin(recommended_ids[:k])]
        return (1 / (recs["popularity_score"] + 1e-9)).mean() if len(recs) > 0 else 0.0

    def evaluate_models_for_city(
        self, city, user_vector, listing_id,
        budget_max=300, room_type="Entire home/apt", top_n=10,
    ):
        popular_ids = self.recommend_popular(
            budget_max=budget_max, city=city, room_type=room_type, top_n=top_n
        )["id"].tolist()

        content_ids = self.recommend_content(
            user_vector, budget_max=budget_max, city=city, room_type=room_type, top_n=top_n
        )["id"].tolist()

        try:
            collab_ids = self.recommend_similar_listings(listing_id, top_n=top_n)["id"].tolist()
        except Exception:
            collab_ids = []

        hybrid_ids = self.hybrid_recommend(
            user_vector=user_vector, listing_id=listing_id,
            budget_max=budget_max, city=city, room_type=room_type, top_n=top_n,
        )["id"].tolist()

        hybrid_time_ids = self.hybrid_time_recommend(
            user_vector=user_vector, listing_id=listing_id,
            budget_max=budget_max, city=city, room_type=room_type, top_n=top_n,
        )["id"].tolist()

        all_id_lists = [popular_ids, content_ids, collab_ids, hybrid_ids, hybrid_time_ids]

        return pd.DataFrame({
            "Model": [
                "Baseline (Popularity)", "Content-Based",
                "Collaborative Filtering", "Hybrid (Static)", "Hybrid (Time-Based)",
            ],
            f"Precision@{top_n}": [self.precision_at_k(ids, k=top_n) for ids in all_id_lists],
            "Coverage (%)":        [self.coverage([ids]) * 100 for ids in all_id_lists],
            f"Diversity@{top_n}":  [self.diversity_at_k(ids, k=top_n) for ids in all_id_lists],
            f"Novelty@{top_n}":    [self.novelty_at_k(ids, k=top_n) for ids in all_id_lists],
        })
