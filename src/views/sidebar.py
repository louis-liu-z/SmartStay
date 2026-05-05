import streamlit as st
from src.constants import CITIES, ROOM_TYPES, TOP_PROPERTY_TYPES, COMMON_AMENITIES


def render_sidebar() -> dict:
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

        property_type = st.selectbox("Property Type", options=TOP_PROPERTY_TYPES, index=1)
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

    return {
        "city": city,
        "budget": budget,
        "room_type": room_type,
        "beds": beds,
        "bathrooms": bathrooms,
        "accommodates": accommodates,
        "min_rating": min_rating,
        "property_type": property_type,
        "amenities": amenities,
        "w_content": w_content,
        "w_collab": w_collab,
        "w_popular": w_popular,
        "w_time": w_time,
        "liked_id_str": liked_id_str,
        "top_n": top_n,
        "gemini_key": gemini_key,
        "find_btn": find_btn,
    }
