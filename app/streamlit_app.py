"""
Starter Streamlit app for Clickstream Behavior Intelligence App.

Run:
    streamlit run app/streamlit_app.py
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Clickstream Behavior Intelligence App",
    page_icon="📊",
    layout="wide",
)


PROCESSED_PATH = Path("data/processed/session_level_dataset.csv")
REPORT_FIGURES_DIR = Path("reports/figures")
REPORT_TABLES_DIR = Path("reports/tables")

CLASSIFICATION_RESULTS_PATH = REPORT_TABLES_DIR / "classification_baseline_results.csv"
CLUSTERING_METRICS_PATH = REPORT_TABLES_DIR / "clustering_baseline_metrics.csv"
CLUSTERING_SIZES_PATH = REPORT_TABLES_DIR / "clustering_cluster_sizes.csv"
CLUSTERING_SILHOUETTE_PATH = REPORT_FIGURES_DIR / "clustering_kmeans_silhouette.png"
CLUSTER_PROFILE_PATH = REPORT_TABLES_DIR / "cluster_profiles_kmeans_k4.csv"
CLUSTER_SIZE_KMEANS_PATH = REPORT_TABLES_DIR / "cluster_sizes_kmeans_k4.csv"
SESSION_ACTION_PATH = REPORT_TABLES_DIR / "session_action_list.csv"
HIGH_ENGAGEMENT_METRICS_PATH = REPORT_TABLES_DIR / "high_engagement_model_metrics.csv"
HIGH_ENGAGEMENT_PRED_PATH = REPORT_TABLES_DIR / "high_engagement_predictions.csv"
HIGH_ENGAGEMENT_TOP_PATH = REPORT_TABLES_DIR / "high_engagement_top50.csv"
HIGH_ENGAGEMENT_MODEL_PATH = Path("models/high_engagement_pipeline.joblib")

HIGH_ENGAGEMENT_DROP = ["session_id", "high_engagement", "total_clicks", "max_order"]


def load_table(path: Path, index_col: int | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=index_col)


def show_image(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.caption(f"Missing figure: {path}")


def build_default_row(feature_df: pd.DataFrame) -> pd.Series:
    defaults = {}
    for col in feature_df.columns:
        series = feature_df[col]
        if pd.api.types.is_numeric_dtype(series):
            defaults[col] = float(series.median()) if not series.dropna().empty else 0.0
        else:
            mode = series.mode(dropna=True)
            defaults[col] = mode.iloc[0] if not mode.empty else ""
    return pd.Series(defaults)


st.title("Clickstream Behavior Intelligence App")
st.caption("Classification and Clustering for User Behavior Analysis")

st.markdown(
    """
    This dashboard is a starter version of the machine learning project.
    The project analyzes user behavior from clickstream data and prepares
    the pipeline for classification and clustering.
    """
)

st.subheader("Project Scope")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Main ML Task 1", "Classification")

with col2:
    st.metric("Main ML Task 2", "Clustering")

with col3:
    st.metric("App Framework", "Streamlit")

st.subheader("Dataset Source")

st.markdown(
    """
    **Clickstream Data for Online Shopping**  
    Source: UCI Machine Learning Repository  
    https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping
    """
)

st.divider()

if PROCESSED_PATH.exists():
    df = pd.read_csv(PROCESSED_PATH)

    st.subheader("Session-Level Dataset Preview")
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Basic Behavioral Statistics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sessions", f"{df['session_id'].nunique():,}")
    c2.metric("Avg Clicks/Session", f"{df['total_clicks'].mean():.2f}")
    c3.metric("Avg Unique Products", f"{df['unique_products'].mean():.2f}")
    c4.metric("Avg Price", f"{df['avg_price'].mean():.2f}")

    if "high_engagement" in df.columns:
        st.subheader("Target Distribution: High Engagement")
        st.bar_chart(df["high_engagement"].value_counts().sort_index())

    if "premium_interest" in df.columns:
        st.subheader("Target Distribution: Premium Interest")
        st.bar_chart(df["premium_interest"].value_counts().sort_index())

    st.divider()

    st.subheader("EDA Highlights")
    event_tab, session_tab = st.tabs(["Event Level", "Session Level"])

    with event_tab:
        left, right = st.columns(2)
        with left:
            show_image(
                REPORT_FIGURES_DIR / "event_price_distribution.png",
                "Price distribution (event level)",
            )
            show_image(
                REPORT_FIGURES_DIR / "event_top_categories.png",
                "Top categories by clicks",
            )
            show_image(
                REPORT_FIGURES_DIR / "event_top_countries.png",
                "Top countries by clicks",
            )
        with right:
            show_image(
                REPORT_FIGURES_DIR / "event_clicks_by_month.png",
                "Clicks by month",
            )
            show_image(
                REPORT_FIGURES_DIR / "event_clicks_by_day.png",
                "Clicks by day",
            )
            show_image(
                REPORT_FIGURES_DIR / "event_clicks_by_page.png",
                "Clicks by page",
            )

        with st.expander("Event-level tables"):
            top_countries = load_table(REPORT_TABLES_DIR / "top_countries.csv", index_col=0)
            if top_countries is not None:
                st.write("Top countries")
                st.dataframe(top_countries, use_container_width=True)

            top_categories = load_table(REPORT_TABLES_DIR / "top_categories.csv", index_col=0)
            if top_categories is not None:
                st.write("Top categories")
                st.dataframe(top_categories, use_container_width=True)

            clicks_by_month = load_table(REPORT_TABLES_DIR / "clicks_by_month.csv", index_col=0)
            if clicks_by_month is not None:
                st.write("Clicks by month")
                st.dataframe(clicks_by_month, use_container_width=True)

            clicks_by_day = load_table(REPORT_TABLES_DIR / "clicks_by_day.csv", index_col=0)
            if clicks_by_day is not None:
                st.write("Clicks by day")
                st.dataframe(clicks_by_day, use_container_width=True)

            clicks_by_page = load_table(REPORT_TABLES_DIR / "clicks_by_page.csv", index_col=0)
            if clicks_by_page is not None:
                st.write("Clicks by page")
                st.dataframe(clicks_by_page, use_container_width=True)

    with session_tab:
        left, right = st.columns(2)
        with left:
            show_image(
                REPORT_FIGURES_DIR / "session_total_clicks_distribution.png",
                "Total clicks per session",
            )
            show_image(
                REPORT_FIGURES_DIR / "session_unique_products_distribution.png",
                "Unique products per session",
            )
        with right:
            show_image(
                REPORT_FIGURES_DIR / "session_avg_price_distribution.png",
                "Average price per session",
            )
            show_image(
                REPORT_FIGURES_DIR / "session_correlation_heatmap.png",
                "Session feature correlation heatmap",
            )

        with st.expander("Session-level tables"):
            session_summary = load_table(
                REPORT_TABLES_DIR / "session_feature_summary.csv", index_col=0
            )
            if session_summary is not None:
                st.write("Session feature summary")
                st.dataframe(session_summary, use_container_width=True)

            session_corr = load_table(
                REPORT_TABLES_DIR / "session_feature_correlation.csv", index_col=0
            )
            if session_corr is not None:
                st.write("Session feature correlation")
                st.dataframe(session_corr, use_container_width=True)

    st.divider()

    st.subheader("Baseline Model Results")
    clf_tab, clust_tab, profile_tab = st.tabs(
        ["Classification", "Clustering", "Cluster Profiling"]
    )

    with clf_tab:
        clf_results = load_table(CLASSIFICATION_RESULTS_PATH)
        if clf_results is None:
            st.info("Run: python scripts/04_baseline_classification.py")
        else:
            st.dataframe(clf_results, use_container_width=True)

    with clust_tab:
        show_image(CLUSTERING_SILHOUETTE_PATH, "KMeans silhouette scores")

        clust_metrics = load_table(CLUSTERING_METRICS_PATH)
        if clust_metrics is None:
            st.info("Run: python scripts/05_baseline_clustering.py")
        else:
            st.write("Clustering metrics")
            st.dataframe(clust_metrics, use_container_width=True)

        clust_sizes = load_table(CLUSTERING_SIZES_PATH)
        if clust_sizes is not None:
            st.write("Cluster sizes")
            st.dataframe(clust_sizes, use_container_width=True)

    with profile_tab:
        profile = load_table(CLUSTER_PROFILE_PATH, index_col=0)
        if profile is None:
            st.info("Run: python scripts/06_build_insight_assets.py")
        else:
            st.write("Cluster profiles (KMeans k=4)")
            st.dataframe(profile, use_container_width=True)

        sizes = load_table(CLUSTER_SIZE_KMEANS_PATH, index_col=0)
        if sizes is not None:
            st.write("Cluster sizes (KMeans k=4)")
            st.dataframe(sizes, use_container_width=True)

    st.divider()

    st.subheader("UX/Product Insights")
    insight_left, insight_right = st.columns(2)

    with insight_left:
        st.markdown("**Peak Activity**")
        clicks_by_month = load_table(REPORT_TABLES_DIR / "clicks_by_month.csv", index_col=0)
        if clicks_by_month is not None and "clicks" in clicks_by_month.columns:
            top_month = clicks_by_month["clicks"].idxmax()
            top_month_clicks = int(clicks_by_month.loc[top_month, "clicks"])
            st.metric("Peak Month", f"{top_month}", f"{top_month_clicks:,} clicks")

        clicks_by_day = load_table(REPORT_TABLES_DIR / "clicks_by_day.csv", index_col=0)
        if clicks_by_day is not None and "clicks" in clicks_by_day.columns:
            top_day = clicks_by_day["clicks"].idxmax()
            top_day_clicks = int(clicks_by_day.loc[top_day, "clicks"])
            st.metric("Peak Day", f"{top_day}", f"{top_day_clicks:,} clicks")

        clicks_by_page = load_table(REPORT_TABLES_DIR / "clicks_by_page.csv", index_col=0)
        if clicks_by_page is not None and "clicks" in clicks_by_page.columns:
            top_page = clicks_by_page["clicks"].idxmax()
            top_page_clicks = int(clicks_by_page.loc[top_page, "clicks"])
            st.metric("Top Page", f"{top_page}", f"{top_page_clicks:,} clicks")

        top_categories = load_table(REPORT_TABLES_DIR / "top_categories.csv", index_col=0)
        if top_categories is not None and "count" in top_categories.columns:
            top_category = top_categories["count"].idxmax()
            top_category_clicks = int(top_categories.loc[top_category, "count"])
            st.metric("Top Category", f"{top_category}", f"{top_category_clicks:,} clicks")

    with insight_right:
        st.markdown("**High Engagement vs Low (mean)**")
        if "high_engagement" in df.columns:
            compare_cols = [
                "total_clicks",
                "unique_products",
                "unique_categories",
                "avg_price",
                "page_diversity_ratio",
            ]
            compare_cols = [col for col in compare_cols if col in df.columns]

            if compare_cols:
                grouped = df.groupby("high_engagement")[compare_cols].mean().T
                grouped.columns = ["low", "high"] if len(grouped.columns) == 2 else grouped.columns
                grouped["delta"] = grouped.get("high", 0) - grouped.get("low", 0)
                st.dataframe(grouped, use_container_width=True)

        st.markdown("**Actionable Recommendations**")
        recs = []
        if clicks_by_page is not None and "clicks" in clicks_by_page.columns:
            recs.append("Prioritize UX improvements on the top page with the most clicks.")
        if top_categories is not None and "count" in top_categories.columns:
            recs.append("Highlight the top category in navigation and promotions.")
        if "high_engagement" in df.columns:
            recs.append("Focus on features that lift unique products and category diversity.")
        if not recs:
            recs.append("Run full EDA and baseline models to unlock insights.")

        for rec in recs:
            st.write(f"- {rec}")

    st.subheader("Action List: Top Sessions")
    action_list = load_table(SESSION_ACTION_PATH)
    if action_list is None:
        st.info("Run: python scripts/06_build_insight_assets.py")
    else:
        st.dataframe(action_list, use_container_width=True)

    st.subheader("Auto Insight Narratives")
    narratives = []
    if action_list is not None and "engagement_score" in action_list.columns:
        top_score = float(action_list["engagement_score"].iloc[0])
        narratives.append(
            f"Top sessions have an engagement score around {top_score:.2f} or higher."
        )

    if profile is not None and "total_clicks" in profile.columns:
        top_cluster = profile["total_clicks"].idxmax()
        top_value = float(profile.loc[top_cluster, "total_clicks"])
        narratives.append(
            f"Cluster {top_cluster} shows the highest average total clicks ({top_value:.2f})."
        )

    if top_categories is not None and "count" in top_categories.columns:
        top_category = top_categories["count"].idxmax()
        narratives.append(f"Category {top_category} dominates click volume.")

    if not narratives:
        narratives.append("Run insight asset builder to generate narratives.")

    for note in narratives:
        st.write(f"- {note}")

    st.divider()

    st.subheader("Prediction: High Engagement")
    saved_tab, manual_tab = st.tabs(["Saved Outputs", "Manual Input"])

    with saved_tab:
        metrics = load_table(HIGH_ENGAGEMENT_METRICS_PATH)
        if metrics is None:
            st.info("Run: python scripts/07_train_high_engagement_model.py")
        else:
            st.write("Model metrics")
            st.dataframe(metrics, use_container_width=True)

        top_predictions = load_table(HIGH_ENGAGEMENT_TOP_PATH)
        if top_predictions is None:
            st.info("Run: python scripts/08_predict_high_engagement.py")
        else:
            st.write("Top predicted sessions")
            st.dataframe(top_predictions, use_container_width=True)

    with manual_tab:
        if not HIGH_ENGAGEMENT_MODEL_PATH.exists():
            st.info("Run: python scripts/07_train_high_engagement_model.py")
        else:
            feature_df = df.drop(columns=[col for col in HIGH_ENGAGEMENT_DROP if col in df.columns])
            default_row = build_default_row(feature_df)

            form_features = [
                "unique_products",
                "unique_categories",
                "unique_pages",
                "avg_price",
                "max_price",
                "min_price",
                "premium_click_ratio",
                "page_diversity_ratio",
                "category_diversity_ratio",
                "product_diversity_ratio",
                "colour_diversity_ratio",
            ]
            form_features = [f for f in form_features if f in feature_df.columns]

            with st.form("prediction_form"):
                st.caption("Inputs not shown here will use dataset medians/modes.")

                inputs = {}
                for col in form_features:
                    default_value = float(default_row[col])
                    if col.endswith("_ratio"):
                        inputs[col] = st.number_input(
                            col,
                            min_value=0.0,
                            max_value=1.0,
                            value=max(0.0, min(1.0, default_value)),
                            step=0.01,
                        )
                    elif col in {"unique_products", "unique_categories", "unique_pages"}:
                        inputs[col] = st.number_input(
                            col,
                            min_value=0,
                            value=int(round(default_value)),
                            step=1,
                        )
                    else:
                        inputs[col] = st.number_input(
                            col,
                            min_value=0.0,
                            value=default_value,
                            step=1.0,
                        )

                submitted = st.form_submit_button("Predict")

            if submitted:
                row = default_row.copy()
                for key, value in inputs.items():
                    row[key] = value

                input_df = pd.DataFrame([row], columns=feature_df.columns)
                pipeline = joblib.load(HIGH_ENGAGEMENT_MODEL_PATH)

                pred_label = int(pipeline.predict(input_df)[0])
                pred_proba = None
                if hasattr(pipeline, "predict_proba"):
                    proba = pipeline.predict_proba(input_df)
                    if proba.shape[1] > 1:
                        pred_proba = float(proba[0, 1])
                elif hasattr(pipeline, "decision_function"):
                    pred_proba = float(pipeline.decision_function(input_df)[0])

                st.subheader("Prediction Result")
                label_text = "High Engagement" if pred_label == 1 else "Low Engagement"
                st.metric("Predicted Label", label_text)
                if pred_proba is not None:
                    st.metric("Predicted Probability", f"{pred_proba:.3f}")

else:
    st.warning(
        """
        Processed dataset not found.

        Please run these commands first:

        1. `python scripts/00_download_data.py`
        2. `python scripts/02_feature_engineering.py`
        """
    )

st.divider()

st.subheader("Next Development Plan")

st.markdown(
    """
    1. Complete EDA notebook.
    2. Add preprocessing pipeline.
    3. Train classification models.
    4. Train clustering models.
    5. Add model comparison dashboard.
    6. Add Docker and cloud deployment workflow.
    """
)
