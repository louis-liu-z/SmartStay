import pandas as pd
import streamlit as st
from src.gemini import gemini_explain


def render_recommendations(gemini_key: str) -> None:
    if st.session_state["results"] is None:
        st.header("Your Recommended Stays")
        st.info("Set your preferences in the sidebar and click **Find Stays**.")
        st.stop()

    results: pd.DataFrame = st.session_state["results"]
    city_label = st.session_state["city"].replace("-", " ").title()

    col_title, col_btn = st.columns([5, 2])
    with col_title:
        st.header("Your Recommended Stays")
    with col_btn:
        st.write("")
        explain_btn = st.button(
            "Explain with Gemini AI",
            use_container_width=True,
            disabled=not gemini_key,
            help="Enter your Google API Key in the sidebar to enable this." if not gemini_key else "",
        )

    if results.empty:
        st.warning(
            f"No listings found in **{city_label}** within your filters. "
            "Try increasing your budget or relaxing other filters."
        )
        return

    if explain_btn:
        with st.spinner("Generating personalized explanations…"):
            st.session_state["explanations"] = gemini_explain(
                results,
                {
                    "budget":    st.session_state["budget"],
                    "city":      st.session_state["city"],
                    "room_type": st.session_state["room_type"],
                    "amenities": st.session_state["amenities_str"],
                },
                gemini_key,
            )

    st.success(f"Found **{len(results)}** stays in **{city_label}** — sorted by hybrid score")

    explanations: list[str] = st.session_state.get("explanations") or []

    score_col = "hybrid_time_score"

    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        score = float(row[score_col])
        score_pct = max(0.0, min(1.0, score)) * 100
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
                f"""
                <div style="padding: 4px 0 8px 0;">
                  <div style="font-size:0.7rem; color:#888; margin-bottom:14px;">Match score</div>
                  <div style="position:relative; height:6px; background:#e9ecef; border-radius:3px; margin-bottom:6px;">
                    <div style="
                      position:absolute;
                      left:calc({score_pct:.2f}% - 6px);
                      top:-13px;
                      width:0; height:0;
                      border-left:6px solid transparent;
                      border-right:6px solid transparent;
                      border-top:10px solid #FF5A5F;">
                    </div>
                    <div style="
                      position:absolute;
                      left:calc({score_pct:.2f}% - 16px);
                      top:-28px;
                      font-size:0.72rem;
                      font-weight:600;
                      color:#FF5A5F;
                      width:32px;
                      text-align:center;">
                      {score:.3f}
                    </div>
                  </div>
                  <div style="display:flex; justify-content:space-between; font-size:0.65rem; color:#bbb;">
                    <span>0</span><span>1</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if rank - 1 < len(explanations):
            st.markdown(
                f'<p style="color:black; margin: 6px 0 4px 0;">{explanations[rank - 1]}</p>',
                unsafe_allow_html=True,
            )

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
