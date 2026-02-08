import streamlit as st
import pandas as pd
import numpy as np
import joblib
import cloudpickle
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import altair as alt

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Credit Card Churn Dashboard", layout="wide")

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model_pipeline.pkl"
THRESH_PATH = APP_DIR / "best_threshold.pkl"

@st.cache_resource
def load_pipeline():
    with MODEL_PATH.open("rb") as f:
        return cloudpickle.load(f)
    
def load_shap_explainer(pipeline):
    """
    Cache SHAP explainer so it is not rebuilt on every rerun.
    """
    model = pipeline.named_steps["model"]
    return shap.TreeExplainer(model)

@st.cache_data
def load_threshold():
    return joblib.load(THRESH_PATH)

pipeline = load_pipeline()
best_threshold = load_threshold()

# Expected columns (raw features your pipeline expects)
REQUIRED_COLS = [
    "Customer_Age", "Gender", "Dependent_count", "Education_Level", "Marital_Status",
    "Income_Category", "Card_Category", "Months_on_book", "Total_Relationship_Count",
    "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal",
    "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1"
]

# --------------------------------
# Retrieve original feature labels
# --------------------------------
categorical_cols = [
    "Gender", "Education_Level", "Marital_Status",
    "Income_Category", "Card_Category"
]

def ohe_rename_map(pipeline, cat_cols):
    """
    Converts one-hot encoded names back to original labels
    E.g. Education_Level_3.0 -> Education_Level=Graduate
    """
    ord_enc = pipeline.named_steps["to_ordinal"].named_transformers_["cat"]
    # Map ordinal-encoded values to original labels (e.g. Education_Level_2.0 → Education_Level_Graduate)
    code_to_label = {
        col: {float(i): str(cat) for i, cat in enumerate(cats)}
        for col, cats in zip(cat_cols, ord_enc.categories_)
    }

    preprocess = pipeline.named_steps["preprocess"]
    ohe = preprocess.named_transformers_["cat_ohe"]

    # Get feature names created by OneHotEncoder.
    ohe_names = ohe.get_feature_names_out(cat_cols)

    def rename_ohe_feature(name: str) -> str:
        """
        Renames one-hot encoded feature names to orginal labels
        using dictionary from code_to_label
        """
        # Split on last underscore so columns like Education_Level work
        # E.g. "Education_Level_1.0" → col="Education_Level", val="1.0"
        col, val = name.rsplit("_", 1)

        try:
            code = float(val)  # "0.0", "1.0", etc.
        except ValueError:
            return name

        label = code_to_label.get(col, {}).get(code, val)
        return f"{col}={label}"

    return rename_ohe_feature

# ------------------------
# Get Feature Importance
# ------------------------
def get_feature_importance_df(pipeline) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    preprocess = pipeline.named_steps["preprocess"]
    feature_names = preprocess.get_feature_names_out()
    importances = model.feature_importances_

    fi = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return fi

# ----------------------------
# UI Header
# ----------------------------
st.markdown("# :credit_card: Credit Card Churn — Deployment Dashboard")
st.caption("Upload a CSV of customers to score churn risk and view key dashboard metrics.")

with st.expander("What columns should my CSV contain?", expanded=False):
    st.write("Your CSV should contain these feature columns:")
    st.code(", ".join(REQUIRED_COLS))

# ----------------------------
# Upload
# ----------------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV to generate predictions and view the dashboard.")
    st.stop()

df = pd.read_csv(uploaded_file)

# Validate required columns
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error("Your CSV is missing required columns:")
    st.write(missing)
    st.stop()

# ----------------------------
# Predict
# ----------------------------
with st.spinner("Scoring customers..."):
    probs = pipeline.predict_proba(df)[:, 1]
    preds = (probs >= best_threshold).astype(int)

df_output = df.copy()
df_output["Churn Probability"] = np.round(probs, 3)
df_output["Prediction"] = np.where(preds == 1, "Churn", "Stay")

# ----------------------------
# Dashboard metrics
# ----------------------------
total = len(df_output)
num_churn = int((df_output["Prediction"] == "Churn").sum())
num_stay = total - num_churn
churn_rate = (num_churn / total) if total else 0.0

avg_prob = float(np.mean(probs)) if total else 0.0
median_prob = float(np.median(probs)) if total else 0.0
p90_prob = float(np.quantile(probs, 0.90)) if total else 0.0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Customers scored", f"{total}")
c2.metric("Predicted churn", f"{num_churn}")
c3.metric("Predicted stay", f"{num_stay}")
c4.metric("Churn rate", f"{churn_rate:.1%}")
c5.metric("Avg churn probability", f"{avg_prob:.3f}")

# ----------------------------
# Charts / tables
# ----------------------------
left, right = st.columns([1.1, 0.9])

with left:
    st.subheader("Churn probability distribution")
    # Streamlit histogram: use st.bar_chart over binned counts
    bins = np.linspace(0, 1, 21)  # 20 bins
    counts, edges = np.histogram(probs, bins=bins)
    hist_df = pd.DataFrame({
        "prob_bin": [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(len(edges)-1)],
        "count": counts
    }).set_index("prob_bin")
    st.bar_chart(hist_df)

    st.caption(f"Median: {median_prob:.3f} | 90th percentile: {p90_prob:.3f}")

with right:
    st.subheader("Highest-risk customers (Top 20)")
    top_n = df_output.sort_values("Churn Probability", ascending=False).head(20)

    # If you have an ID column like CLIENTNUM, show it first; otherwise show the table as-is
    cols = top_n.columns.tolist()
    preferred_id_cols = [c for c in ["CLIENTNUM", "Customer_ID", "ID"] if c in cols]
    ordered_cols = preferred_id_cols + [c for c in cols if c not in preferred_id_cols]

    st.dataframe(top_n[ordered_cols], use_container_width=True)

# Full scored dataset
st.subheader("Scored dataset")
st.dataframe(df_output, use_container_width=True)

# Download results
csv = df_output.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Results (CSV)",
    data=csv,
    file_name="churn_predictions.csv",
    mime="text/csv"
)

# ----------------------------
# Feature Importance
# ----------------------------
st.subheader("Model Insights")

with st.expander("Feature importance (global)", expanded=False):
    try:
        fi = get_feature_importance_df(pipeline)

        rename_ohe_feature = ohe_rename_map(pipeline, categorical_cols)

        # Apply renaming to categorical OHE features only; leave numeric alone
        def prettify(name: str) -> str:
            """
            Apply "rename_ohe_feature" function defined above
            to rename one-hot encoded features to original labels
            """
            for col in categorical_cols:
                # If categorical column is "example_0.0", rename it
                if name.startswith(col + "_"):
                    return rename_ohe_feature(name)
            return name

        fi["feature_readable"] = fi["feature"].apply(prettify)

        top_k = st.slider("Show top N features", 5, 40, 10, key="top_k_raw")
        df_display = (
            fi[["feature_readable", "importance"]]
            .head(top_k)
            .reset_index(drop=True)
        )
        # Ensures index starts from 1 in the feature importance table
        df_display.index = df_display.index + 1
        st.dataframe(df_display, use_container_width=True)
        
        # Take top_k rows and reverse order so highest importance is at the top
        fi_plot = fi[["feature_readable", "importance"]].head(top_k)
        chart = (
            alt.Chart(fi_plot)
            .mark_bar()
            .encode(
                x=alt.X("importance:Q", title="Feature importance"),
                y=alt.Y(
                    "feature_readable:N",
                    sort=None,
                    title=""
                    ),
                tooltip=["feature_readable", "importance"]
                )
            .properties(height=30 * len(fi_plot))  # auto height
            )
        
        st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.warning("Could not extract feature importance from the loaded pipeline.")
        st.exception(e)
        
with st.expander("SHAP summary plots", expanded=False):
    try:
        feature_creator = pipeline.named_steps["feature_creator"]
        to_ordinal = pipeline.named_steps["to_ordinal"]
        preprocess = pipeline.named_steps["preprocess"]
        model = pipeline.named_steps["model"]

        # ---- Build model-feature matrix from uploaded df ----
        # Keep all columns; preprocess will select what it needs.
        X_feature_create = feature_creator.transform(df)
        X_ord = to_ordinal.transform(X_feature_create)
        X_model = preprocess.transform(X_ord)

        feature_names = preprocess.get_feature_names_out()

        # Convert to DataFrame so we can rename columns nicely for plots
        X_model_df = pd.DataFrame(X_model, columns=feature_names)

        # rename one-hot encoded names to original labels
        rename_ohe_feature = ohe_rename_map(pipeline, categorical_cols)
        def prettify(name: str) -> str:
            """
            Apply "rename_ohe_feature" function defined above
            to rename one-hot encoded features to original labels
            """
            for col in categorical_cols:
                # If categorical column is "example_0.0", rename it
                if name.startswith(col + "_"):
                    return rename_ohe_feature(name)
            return name
        X_model_df = X_model_df.rename(columns=lambda c: prettify(c))

        # ---- SHAP explainer ----
        # For XGBoost sklearn wrapper, TreeExplainer usually works well
        # Build SHAP explanation
        explainer = load_shap_explainer(pipeline)

        # Compute SHAP values
        shap_values = explainer(X_model_df)

        # Binary classification: some versions return list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 = churn

        st.caption("Beeswarm: each dot is a customer; right = increases churn probability, left = decreases.")

        # ---- Beeswarm plot ----
        fig1, ax1 = plt.subplots()
        shap.plots.beeswarm(shap_values, max_display=20)
        st.pyplot(fig1, clear_figure=True)

        st.caption("Bar: mean(|SHAP|) global importance across sampled customers.")

        # ---- Bar plot ----
        fig2, ax2 = plt.subplots()
        shap.plots.bar(shap_values, max_display=20)
        st.pyplot(fig2, clear_figure=True)

    except Exception as e:
        st.warning("Could not generate SHAP summary plots.")
        st.exception(e)