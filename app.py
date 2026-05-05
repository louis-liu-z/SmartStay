import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ── locate project folder so imports work whether launched from anywhere ──────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from recommender import (
    SmartStayRecommender,
    CITIES,
    ROOM_TYPES,
    TOP_PROPERTY_TYPES,
    COMMON_AMENITIES,
)

DATA_PATH = os.path.join(APP_DIR, "clean_ca_df.csv")

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartStay — California Airbnb Recommender",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .listing-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border-left: 4px solid #FF5A5F;
    }
    .score-label { font-size: 0.75rem; color: #888; margin-bottom: 2px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── load model (cached — built only once per session) ─────────────────────────
@st.cache_resource(show_spinner="Building SmartStay model — one-time setup, ~30 s…")
def load_recommender():
    return SmartStayRecommender(data_path=DATA_PATH)


rec = load_recommender()

for _key in ("results", "user_vec", "listing_id", "city", "budget", "room_type", "amenities_str", "eval_df"):
    if _key not in st.session_state:
        st.session_state[_key] = None

# ── Gemini helper ─────────────────────────────────────────────────────────────
def gemini_explain(listings_df: pd.DataFrame, user_prefs: dict, api_key: str) -> str:
    try:
        import google.generativeai as genai
    except ImportError:
        return "google-generativeai is not installed. Run: pip install google-generativeai"

    genai.configure(api_key=api_key)
    try:
        model = next(
            (genai.GenerativeModel(m.name)
             for m in genai.list_models()
             if "generateContent" in m.supported_generation_methods),
            None,
        )
    except Exception as e:
        return f"Could not connect to Gemini: {e}"

    if model is None:
        return "No Gemini model found that supports generateContent."

    details = []
    for i, (_, row) in enumerate(listings_df.iterrows()):
        amenities_preview = str(row.get("amenities_clean", ""))[:200]
        details.append(
            f"Listing {i + 1}:\n"
            f"- Name: {row['name']}\n"
            f"- City: {row['City']}\n"
            f"- Room type: {row['room_type']}\n"
            f"- Price: ${row['price']:.0f}/night\n"
            f"- Rating: {row['review_scores_rating']}/5\n"
            f"- Amenities: {amenities_preview}"
        )

    prompt = f"""
A user is looking for an Airbnb in California with these preferences:
- Budget: ${user_prefs.get('budget')} per night
- Location: {user_prefs.get('city')}
- Room type: {user_prefs.get('room_type')}
- Must-have amenities: {user_prefs.get('amenities')}

Here are {len(listings_df)} recommended listings:
{chr(10).join(details)}

For each listing, write a concise 2-3 sentence explanation of why it matches the user's needs.
Be specific — mention price, rating, or amenities that align with their preferences.
Present the explanations as a numbered list matching the order above.
"""
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Gemini error: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🏠 SmartStay")
    st.caption("California Airbnb Recommender")
    st.divider()

    st.subheader("Where & When")
    city = st.selectbox(
        "City",
        options=CITIES,
        format_func=lambda x: x.replace("-", " ").title(),
    )
    budget = st.slider("Max Budget ($/night)", min_value=20, max_value=2000,
                       value=300, step=10)
    room_type = st.selectbox("Room Type", options=ROOM_TYPES)

    st.divider()
    st.subheader("Stay Details")
    col_a, col_b = st.columns(2)
    with col_a:
        beds = st.number_input("Beds", min_value=1, max_value=10, value=2)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=1)
    with col_b:
        accommodates = st.number_input("Guests", min_value=1, max_value=16, value=4)
        min_rating = st.number_input("Min Rating", min_value=1.0, max_value=5.0,
                                     value=4.5, step=0.1, format="%.1f")

    property_type = st.selectbox("Property Type", options=TOP_PROPERTY_TYPES,
                                  index=1)  # default: Entire rental unit
    amenities = st.multiselect(
        "Must-have Amenities",
        options=COMMON_AMENITIES,
        default=["wifi", "kitchen"],
    )

    st.divider()
    with st.expander("Advanced: Model Weights"):
        st.caption("Weights are auto-normalized to sum to 1.")
        w_content = st.slider("Content (your preferences)", 0.0, 1.0, 0.40, 0.05)
        w_collab  = st.slider("Collaborative (similar listings)", 0.0, 1.0, 0.25, 0.05)
        w_popular = st.slider("Popularity (reviews & rating)", 0.0, 1.0, 0.20, 0.05)
        w_time    = st.slider("Recency (activity signal)", 0.0, 1.0, 0.15, 0.05)
        total_w   = w_content + w_collab + w_popular + w_time
        st.caption(f"Raw total: {total_w:.2f}")
        liked_id_str = st.text_input(
            "Liked Listing ID",
            placeholder="Optional — activates CF signal",
            help="Enter a listing ID you liked to boost similar listings.",
        )
        top_n = st.slider("Results to show", 5, 20, 10)

    with st.expander("AI Explanations (Gemini)"):
        gemini_key = st.text_input("Google API Key", type="password",
                                   help="Required to generate AI explanations")

    st.divider()
    find_btn = st.button("Find Stays", type="primary", use_container_width=True)


# ── run recommendation on button press ────────────────────────────────────────
if find_btn:
    amenities_str = " ".join(amenities)

    listing_id = None
    if liked_id_str.strip():
        try:
            listing_id = int(liked_id_str.strip())
        except ValueError:
            st.sidebar.warning("Listing ID must be a number — ignoring.")

    # normalize weights
    total_w = w_content + w_collab + w_popular + w_time
    if total_w == 0:
        total_w = 1.0
    wc, wcf, wp, wt = (
        w_content / total_w, w_collab / total_w,
        w_popular / total_w, w_time / total_w,
    )

    with st.spinner("Finding your perfect stays…"):
        user_vec = rec.build_user_vector(
            price=budget, beds=beds, bathrooms=bathrooms,
            accommodates=accommodates, min_rating=min_rating,
            room_type=room_type, property_type=property_type,
            amenities_str=amenities_str,
        )
        results = rec.hybrid_time_recommend(
            user_vector=user_vec,
            listing_id=listing_id,
            budget_max=budget,
            city=city,
            room_type=room_type,
            w_content=wc, w_collab=wcf, w_popular=wp, w_time=wt,
            top_n=top_n,
        )

    st.session_state.update(
        results=results,
        user_vec=user_vec,
        listing_id=listing_id,
        city=city,
        budget=budget,
        room_type=room_type,
        amenities_str=amenities_str,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_rec, tab_eda, tab_compare = st.tabs(
    ["Recommendations", "Explore Data", "Compare Models"]
)

# ── TAB 1: Recommendations ────────────────────────────────────────────────────
with tab_rec:
    st.header("Your Recommended Stays")

    if st.session_state["results"] is None:
        st.info("Set your preferences in the sidebar and click **Find Stays**.")
        st.stop()

    results: pd.DataFrame = st.session_state["results"]
    city_label = st.session_state["city"].replace("-", " ").title()

    if results.empty:
        st.warning(
            f"No listings found in **{city_label}** within your filters. "
            "Try increasing your budget or relaxing other filters."
        )
    else:
        st.success(f"Found **{len(results)}** stays in **{city_label}** — sorted by hybrid score")

        # Score range for progress bar normalization
        score_col = "hybrid_time_score"
        score_min = float(results[score_col].min())
        score_max = float(results[score_col].max())
        score_range = score_max - score_min if score_max > score_min else 1.0

        for rank, (_, row) in enumerate(results.iterrows(), start=1):
            score = float(row[score_col])
            bar_val = (score - score_min) / score_range
            rating = row["review_scores_rating"]
            rating_str = f"{rating:.1f}/5" if pd.notna(rating) else "N/A"
            amenities_preview = str(row.get("amenities_clean", ""))
            amenities_preview = amenities_preview[:120] + "…" if len(amenities_preview) > 120 else amenities_preview

            st.markdown(
                f"""<div class="listing-card">
                <strong>#{rank} — {row['name']}</strong><br>
                <span style="color:#666">{row['City'].replace('-',' ').title()} &nbsp;·&nbsp; {row['room_type']}</span>
                </div>""",
                unsafe_allow_html=True,
            )

            col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
            col1.metric("Price", f"${row['price']:.0f}/night")
            col2.metric("Rating", rating_str)
            col3.metric("Listing ID", int(row["id"]))
            with col4:
                st.markdown(
                    f'<p class="score-label">Match score: {score:.4f}</p>',
                    unsafe_allow_html=True,
                )
                st.progress(bar_val)
            if amenities_preview:
                st.caption(f"Amenities: {amenities_preview}")
            st.divider()

        # ── CSV download ──────────────────────────────────────────────────────
        csv_bytes = results.drop(columns=["amenities_clean"], errors="ignore").to_csv(
            index=False
        ).encode()
        st.download_button(
            "Download results as CSV",
            data=csv_bytes,
            file_name=f"smartstay_{st.session_state['city']}.csv",
            mime="text/csv",
        )

        # ── Gemini AI explanations ────────────────────────────────────────────
        st.subheader("AI Explanations")
        if st.button("Explain these recommendations with Gemini AI"):
            if not gemini_key:
                st.error(
                    "Enter your Google API Key in the sidebar under "
                    "**AI Explanations (Gemini)** to use this feature."
                )
            else:
                with st.spinner("Generating personalized explanations…"):
                    explanation = gemini_explain(
                        results,
                        {
                            "budget":    st.session_state["budget"],
                            "city":      st.session_state["city"],
                            "room_type": st.session_state["room_type"],
                            "amenities": st.session_state["amenities_str"],
                        },
                        gemini_key,
                    )
                st.markdown(explanation)


# ── TAB 2: Explore Data ───────────────────────────────────────────────────────
with tab_eda:
    st.header("Explore the Dataset")
    raw = rec.raw

    col1, col2 = st.columns(2)

    with col1:
        price_cap = raw["price"].quantile(0.99)
        fig = px.histogram(
            raw[raw["price"] <= price_cap],
            x="price", nbins=60,
            title="Price Distribution (capped at 99th percentile)",
            labels={"price": "Price ($/night)"},
            color_discrete_sequence=["#FF5A5F"],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        rt = raw["room_type"].value_counts().reset_index()
        rt.columns = ["Room Type", "Count"]
        fig = px.bar(
            rt, x="Room Type", y="Count",
            title="Room Type Distribution",
            color="Room Type",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        city_counts = raw["City"].value_counts().reset_index()
        city_counts.columns = ["City", "Count"]
        city_counts["City"] = city_counts["City"].str.replace("-", " ").str.title()
        fig = px.bar(
            city_counts, x="City", y="Count",
            title="Listings by City",
            color="City",
        )
        fig.update_xaxes(tickangle=35)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        prop = raw["property_type"].value_counts().head(12).reset_index()
        prop.columns = ["Property Type", "Count"]
        fig = px.bar(
            prop, x="Count", y="Property Type",
            orientation="h",
            title="Top 12 Property Types",
            color="Count",
            color_continuous_scale="Reds",
        )
        fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    # Rating distribution by city
    st.subheader("Rating Distribution by City")
    rating_city = raw.dropna(subset=["review_scores_rating", "City"])
    rating_city = rating_city[rating_city["City"].isin(CITIES)]
    rating_city["City_label"] = rating_city["City"].str.replace("-", " ").str.title()
    fig = px.box(
        rating_city, x="City_label", y="review_scores_rating",
        title="Review Score Distribution by City",
        labels={"City_label": "City", "review_scores_rating": "Rating"},
        color="City_label",
    )
    fig.update_xaxes(tickangle=30)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ── TAB 3: Compare Models ─────────────────────────────────────────────────────
with tab_compare:
    st.header("Compare Recommendation Models")
    st.caption(
        "Evaluate all five models on the same filters: "
        "Precision@K, Coverage, Diversity, and Novelty."
    )

    if st.session_state["user_vec"] is None:
        st.info("Run a search first (click **Find Stays**) to enable model comparison.")
    else:
        eval_city = st.selectbox(
            "City to evaluate",
            options=CITIES,
            format_func=lambda x: x.replace("-", " ").title(),
            index=CITIES.index(st.session_state["city"])
            if st.session_state["city"] in CITIES else 0,
            key="eval_city",
        )
        eval_budget = st.number_input(
            "Budget for evaluation ($/night)", 50, 2000,
            value=int(st.session_state["budget"]), step=10, key="eval_budget",
        )
        eval_room = st.selectbox(
            "Room type for evaluation", ROOM_TYPES,
            index=ROOM_TYPES.index(st.session_state["room_type"])
            if st.session_state["room_type"] in ROOM_TYPES else 0,
            key="eval_room",
        )

        if st.button("Run Comparison", type="primary"):
            lid = st.session_state["listing_id"] or int(rec.raw.iloc[0]["id"])
            with st.spinner("Evaluating all 5 models…"):
                eval_df = rec.evaluate_models_for_city(
                    city=eval_city,
                    user_vector=st.session_state["user_vec"],
                    listing_id=lid,
                    budget_max=eval_budget,
                    room_type=eval_room,
                    top_n=10,
                )
            st.session_state["eval_df"] = eval_df

        if "eval_df" in st.session_state:
            eval_df = st.session_state["eval_df"]

            # Styled table
            metric_cols = [c for c in eval_df.columns if c != "Model"]
            styled = eval_df.style.format(
                {c: "{:.4f}" for c in metric_cols}
            ).background_gradient(subset=metric_cols, cmap="YlGn")
            st.dataframe(styled, use_container_width=True)

            # One chart per metric
            st.subheader("Visual Comparison")
            for metric in metric_cols:
                fig = px.bar(
                    eval_df, x="Model", y=metric,
                    title=metric,
                    color="Model",
                    text_auto=".3f",
                )
                fig.update_layout(showlegend=False, xaxis_tickangle=20)
                st.plotly_chart(fig, use_container_width=True)
