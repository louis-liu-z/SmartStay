import pandas as pd
import plotly.express as px
import streamlit as st
from src.constants import CITIES


def render_explore(raw: pd.DataFrame) -> None:
    st.header("Explore the Dataset")

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
        fig = px.bar(rt, x="Room Type", y="Count", title="Room Type Distribution", color="Room Type")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        city_counts = raw["City"].value_counts().reset_index()
        city_counts.columns = ["City", "Count"]
        city_counts["City"] = city_counts["City"].str.replace("-", " ").str.title()
        fig = px.bar(city_counts, x="City", y="Count", title="Listings by City", color="City")
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

    st.subheader("Rating Distribution by City")
    rating_city = raw.dropna(subset=["review_scores_rating", "City"])
    rating_city = rating_city[rating_city["City"].isin(CITIES)]
    rating_city = rating_city.copy()
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
