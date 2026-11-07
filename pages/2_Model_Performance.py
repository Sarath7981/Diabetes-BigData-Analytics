import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Model Performance", page_icon="🤖", layout="wide")
st.title("Machine Learning Model Performance")
st.write("Comparison of Logistic Regression, Random Forest and Gradient Boosting.")

@st.cache_data
def load_results():
    return pd.read_csv("results/model_comparison.csv")

df = load_results()
best_f1 = df.loc[df["F1_Score"].idxmax()]
best_accuracy = df.loc[df["Accuracy"].idxmax()]
best_recall = df.loc[df["Recall"].idxmax()]
best_auc = df.loc[df["ROC_AUC"].idxmax()]
fastest = df.loc[df["Training_Time_Seconds"].idxmin()]

c1, c2, c3 = st.columns(3)
c1.metric("Best F1-score", f'{best_f1["F1_Score"]:.2%}', best_f1["Model"])
c2.metric("Best Accuracy", f'{best_accuracy["Accuracy"]:.2%}', best_accuracy["Model"])
c3.metric("Best Recall", f'{best_recall["Recall"]:.2%}', best_recall["Model"])
c4, c5 = st.columns(2)
c4.metric("Best ROC-AUC", f'{best_auc["ROC_AUC"]:.4f}', best_auc["Model"])
c5.metric("Fastest training", f'{fastest["Training_Time_Seconds"]:.2f} sec', fastest["Model"])

st.header("Complete model comparison")
display = df.copy()
for col in ["Accuracy", "Precision", "Recall", "F1_Score"]:
    display[col] *= 100
st.dataframe(display.round(2), use_container_width=True, hide_index=True)

for metric, title in [
    ("Accuracy", "Accuracy comparison"),
    ("Precision", "Precision comparison"),
    ("Recall", "Recall comparison"),
    ("F1_Score", "F1-score comparison"),
    ("ROC_AUC", "ROC-AUC comparison"),
    ("Training_Time_Seconds", "Training time comparison")
]:
    st.subheader(title)
    plot_df = df[["Model", metric]].copy()
    if metric in ["Accuracy", "Precision", "Recall", "F1_Score"]:
        plot_df[metric] *= 100
    fig = px.bar(plot_df, x="Model", y=metric, color="Model", text=metric)
    st.plotly_chart(fig, use_container_width=True)

st.success(f'Random Forest was selected by F1-score: {best_f1["F1_Score"]:.2%}.')
st.caption("Metrics are from the held-out test split. This educational model is not a medical diagnostic tool.")
st.download_button("Download model comparison CSV", df.to_csv(index=False), "model_comparison.csv", "text/csv")
