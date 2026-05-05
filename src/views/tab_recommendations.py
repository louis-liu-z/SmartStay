import pandas as pd
import streamlit as st
from src.gemini import gemini_explain


def render_recommendations(gemini_key: str) -> None:
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
        return

    st.success(f"Found **{len(results)}** stays in **{city_label}** — sorted by hybrid score")

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
        amenities_preview = (
            amenities_preview[:120] + "…" if len(amenities_preview) > 120 else amenities_preview
        )

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

    csv_bytes = results.drop(columns=["amenities_clean"], errors="ignore").to_csv(
        index=False
    ).encode()
    st.download_button(
        "Download results as CSV",
        data=csv_bytes,
        file_name=f"smartstay_{st.session_state['city']}.csv",
        mime="text/csv",
    )

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
