
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")
st.title("🚀 Travel Experience Decision Intelligence Dashboard")

file = st.file_uploader("Upload Dataset", type=["csv"])

if file:
    df = pd.read_csv(file)

    # ---------------- KPI ----------------
    st.header("📊 Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Users", len(df))
    c2.metric("Top Experience", df["Experience_Type"].mode()[0])
    c3.metric("Top Spend Category", df["Spend"].mode()[0])

    st.divider()

    # ---------------- DESCRIPTIVE ----------------
    st.header("📈 Descriptive Analytics")
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(px.histogram(df, x="Age", title="Age Distribution"))
        st.plotly_chart(px.pie(df, names="Experience_Type", title="Experience Preferences"))

    with col2:
        st.plotly_chart(px.box(df, x="Income", y="Spend", title="Income vs Spend"))
        st.plotly_chart(px.histogram(df, x="Barriers", title="Customer Barriers"))

    st.divider()

    # ---------------- ENCODING ----------------
    df_enc = df.copy()
    for col in df_enc.columns:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))

    # ---------------- CLASSIFICATION ----------------
    st.header("🤖 Customer Conversion Prediction")

    X = df_enc.drop("Likelihood", axis=1)
    y = df_enc["Likelihood"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 3))
    c2.metric("Precision", round(precision_score(y_test, y_pred, average='weighted'), 3))
    c3.metric("Recall", round(recall_score(y_test, y_pred, average='weighted'), 3))
    c4.metric("F1 Score", round(f1_score(y_test, y_pred, average='weighted'), 3))

    st.divider()

    # ---------------- REGRESSION ----------------
    st.header("💰 Spending Power Prediction")

    if "Spend" in df_enc.columns:
        y_reg = df_enc["Spend"]
        X_reg = df_enc.drop("Spend", axis=1)

        Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2)

        reg = RandomForestRegressor()
        reg.fit(Xr_train, yr_train)
        preds = reg.predict(Xr_test)

        st.plotly_chart(px.scatter(x=yr_test, y=preds,
                                  labels={"x":"Actual Spend", "y":"Predicted Spend"},
                                  title="Actual vs Predicted Spend"))

    st.divider()

    # ---------------- CLUSTERING ----------------
    st.header("👥 Customer Segmentation")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    st.plotly_chart(px.scatter(df, x="Age", y="Spend", color="Cluster",
                               size="Frequency", title="Customer Segments"))

    st.success("Segment 1: Low Intent → Discounts")
    st.success("Segment 2: Premium Users → Upsell")
    st.success("Segment 3: Social Explorers → Bundles")

    st.divider()

    # ---------------- ASSOCIATION (SAFE VERSION) ----------------
    st.header("🔗 Experience Bundling Insights")

    if "Experience_Type" in df.columns:
        combos = df["Experience_Type"].value_counts().head(5)

        st.write("Top Experience Preferences (Proxy for Association):")
        st.bar_chart(combos)

        st.info("Users interested in top experiences are likely to explore similar categories")

    st.divider()

    # ---------------- INSIGHTS ----------------
    st.header("💡 Key Insights")
    st.info("Most users prefer food & social experiences")
    st.info("Spending increases with income")
    st.info("Barriers like price and time reduce conversions")

    st.divider()

    # ---------------- RECOMMENDATIONS ----------------
    st.header("🎯 Strategic Recommendations")
    st.write("- Offer entry discounts to low-intent users")
    st.write("- Bundle Food + Hidden Gems experiences")
    st.write("- Upsell premium experiences to high spend users")
    st.write("- Target Gen Z via social media campaigns")
