import os
import sys

import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from src.recommender import SmartStayRecommender
from src.views.sidebar import render_sidebar
from src.views.tab_recommendations import render_recommendations
from src.views.tab_explore import render_explore
from src.views.tab_compare import render_compare

DATA_PATH = os.path.join(APP_DIR, "clean_ca_df.csv")

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


@st.cache_resource(show_spinner="Building SmartStay model — one-time setup, ~30 s…")
def load_recommender():
    return SmartStayRecommender(data_path=DATA_PATH)


rec = load_recommender()

for _key in ("results", "user_vec", "listing_id", "city", "budget", "room_type", "amenities_str", "eval_df"):
    if _key not in st.session_state:
        st.session_state[_key] = None

inputs = render_sidebar()

if inputs["find_btn"]:
    amenities_str = " ".join(inputs["amenities"])

    listing_id = None
    if inputs["liked_id_str"].strip():
        try:
            listing_id = int(inputs["liked_id_str"].strip())
        except ValueError:
            st.sidebar.warning("Listing ID must be a number — ignoring.")

    total_w = inputs["w_content"] + inputs["w_collab"] + inputs["w_popular"] + inputs["w_time"]
    if total_w == 0:
        total_w = 1.0
    wc  = inputs["w_content"] / total_w
    wcf = inputs["w_collab"]  / total_w
    wp  = inputs["w_popular"] / total_w
    wt  = inputs["w_time"]    / total_w

    with st.spinner("Finding your perfect stays…"):
        user_vec = rec.build_user_vector(
            price=inputs["budget"],
            beds=inputs["beds"],
            bathrooms=inputs["bathrooms"],
            accommodates=inputs["accommodates"],
            min_rating=inputs["min_rating"],
            room_type=inputs["room_type"],
            property_type=inputs["property_type"],
            amenities_str=amenities_str,
        )
        results = rec.hybrid_time_recommend(
            user_vector=user_vec,
            listing_id=listing_id,
            budget_max=inputs["budget"],
            city=inputs["city"],
            room_type=inputs["room_type"],
            w_content=wc, w_collab=wcf, w_popular=wp, w_time=wt,
            top_n=inputs["top_n"],
        )

    st.session_state.update(
        results=results,
        user_vec=user_vec,
        listing_id=listing_id,
        city=inputs["city"],
        budget=inputs["budget"],
        room_type=inputs["room_type"],
        amenities_str=amenities_str,
    )

tab_rec, tab_eda, tab_compare = st.tabs(["Recommendations", "Explore Data", "Compare Models"])

with tab_rec:
    render_recommendations(gemini_key=inputs["gemini_key"])

with tab_eda:
    render_explore(rec.raw)

with tab_compare:
    render_compare(rec)
