# app.py — Data Analysis Focused Version

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from io import BytesIO

st.set_page_config(layout="wide", page_title="Healthcare Brand Health Analysis")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data(path="drug_reviews_processed.csv"):
    df = pd.read_csv(path, low_memory=False)
    return df

df = load_data()
if df is None or df.empty:
    st.error("Could not load processed dataset. Please check your file path.")
    st.stop()

# -------------------------------
# Dashboard Header
# -------------------------------
st.title("💊 Healthcare Brand Health & Patient Sentiment Analysis")
st.markdown("""
Analyzed patient feedback on drugs and conditions to understand **brand perception**, **sentiment**, and **effectiveness**.
Data Source: [Kaggle Drug Review Dataset](https://www.kaggle.com/datasets/matiflatif/drugs-review-dataset)""")

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews", f"{len(df):,}")
col2.metric("Unique Drugs", f"{df['urlDrugName'].nunique():,}")
col3.metric("Conditions Covered", f"{df['condition'].nunique():,}")
col4.metric("Avg Rating", f"{df['rating'].mean():.2f} / 10")

st.markdown("---")

# -------------------------------
# 1️⃣ Sentiment Distribution
# -------------------------------
st.header("📊 Sentiment Distribution by Review Type")

sent_cols = {
    "benefitsReview_vader_compound": "Benefits",
    "sideEffectsReview_vader_compound": "Side Effects",
    "commentsReview_vader_compound": "Comments",
    "all_reviews_vader_compound": "Overall"
}

available_cols = [c for c in sent_cols.keys() if c in df.columns]
if available_cols:
    sent_df = df[available_cols].melt(var_name="Aspect", value_name="Sentiment")
    sent_df["Aspect"] = sent_df["Aspect"].map(sent_cols)
    fig = px.box(sent_df, x="Aspect", y="Sentiment", color="Aspect",
                 title="Sentiment Polarity Distribution Across Review Aspects",
                 color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No sentiment columns found in dataset.")

st.markdown("---")

# -------------------------------
# 2️⃣ Brand Health by Drug
# -------------------------------
st.header("🏥 Brand Health Overview by Drug")

drug_summary = df.groupby("urlDrugName").agg({
    "rating": "mean",
    "satisfied": "mean",
    "effectiveness_mapped": "mean",
    "sideEffects_mapped": "mean"
}).reset_index()

drug_summary["satisfied_pct"] = (drug_summary["satisfied"] * 100).round(1)
drug_summary = drug_summary.sort_values("satisfied_pct", ascending=False)

top_drugs = st.slider("Select number of top drugs to display:", 5, 30, 10)

fig2 = px.bar(drug_summary.head(top_drugs),
              x="urlDrugName", y="satisfied_pct",
              color="effectiveness_mapped",
              color_continuous_scale="Greens",
              title=f"Top {top_drugs} Drugs by Satisfaction (%)")
fig2.update_layout(xaxis_tickangle=45)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# -------------------------------
# 3️⃣ Word Cloud Visualization
# -------------------------------
st.header("💬 Word Cloud Comparison")

colA, colB = st.columns(2)

def make_wordcloud(text, colormap):
    wc = WordCloud(width=800, height=400, background_color='white',
                   stopwords=STOPWORDS, colormap=colormap, max_words=200)
    wc.generate(text)
    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    buf.seek(0)
    return buf

sat_text = " ".join(df[df['satisfied']==1]['all_reviews_clean'].dropna().tolist())
unsat_text = " ".join(df[df['satisfied']==0]['all_reviews_clean'].dropna().tolist())

with colA:
    st.subheader("Satisfied Reviews")
    st.image(make_wordcloud(sat_text, "Greens"))

with colB:
    st.subheader("Not Satisfied Reviews")
    st.image(make_wordcloud(unsat_text, "Reds"))

st.markdown("---")

# -------------------------------
# 4️⃣ Effectiveness vs Side Effects
# -------------------------------
st.header("⚖️ Effectiveness vs Side Effects Analysis")

fig3 = px.scatter(df, x="effectiveness_mapped", y="sideEffects_mapped",
                  color="satisfied",
                  color_discrete_map={1: "green", 0: "red"},
                  hover_data=["urlDrugName", "rating"],
                  title="Patient Satisfaction Based on Effectiveness vs Side Effects")
fig3.update_traces(marker=dict(size=8, opacity=0.6))
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# -------------------------------
# 5️⃣ Condition-wise Analysis
# -------------------------------
st.header("🩺 Condition-wise Average Sentiment & Rating")

condition_summary = df.groupby("condition").agg({
    "rating": "mean",
    "satisfied": "mean",
    "all_reviews_vader_compound": "mean"
}).reset_index().sort_values("satisfied", ascending=False)

top_conditions = st.slider("Select number of top conditions to display:", 5, 25, 10)

fig4 = px.bar(condition_summary.head(top_conditions),
              x="condition", y="satisfied",
              color="rating", color_continuous_scale="Blues",
              title="Top Conditions by Satisfaction & Rating")
fig4.update_layout(xaxis_tickangle=45)
st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

st.caption("📘 Built for data-driven storytelling: uncover patient sentiment, treatment perception, and brand reputation in healthcare.")
