import plotly.express as px
import streamlit as st
from src.constants import CITIES, ROOM_TYPES


def render_compare(rec) -> None:
    st.header("Compare Recommendation Models")
    st.caption(
        "Evaluate all five models on the same filters: "
        "Precision@K, Coverage, Diversity, and Novelty."
    )

    if st.session_state["user_vec"] is None:
        st.info("Run a search first (click **Find Stays**) to enable model comparison.")
        return

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

    if st.session_state.get("eval_df") is not None:
        eval_df = st.session_state["eval_df"]

        metric_cols = [c for c in eval_df.columns if c != "Model"]
        styled = eval_df.style.format(
            {c: "{:.4f}" for c in metric_cols}
        ).background_gradient(subset=metric_cols, cmap="YlGn")
        st.dataframe(styled, use_container_width=True)

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
