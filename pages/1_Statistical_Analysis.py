import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Statistical Analysis", page_icon="📊", layout="wide")
st.title("Diabetes Dataset — Statistical Analysis")
st.write("Explore the cleaned CDC Diabetes Health Indicators dataset.")

@st.cache_data
def load_data():
    return pd.read_parquet("data/processed/diabetes_parquet.parquet")

df = load_data()
st.sidebar.header("Filters")
age_values = sorted(df["Age"].dropna().unique().tolist())
selected_ages = st.sidebar.multiselect("Age categories", age_values, default=age_values)
sex_values = sorted(df["Sex"].dropna().unique().tolist())
selected_sex = st.sidebar.multiselect("Sex category", sex_values, default=sex_values)
filtered = df[df["Age"].isin(selected_ages) & df["Sex"].isin(selected_sex)]

total = len(filtered)
positive = int((filtered["Diabetes_binary"] == 1).sum())
negative = int((filtered["Diabetes_binary"] == 0).sum())
a, b, c, d = st.columns(4)
a.metric("Records", f"{total:,}")
b.metric("Input features", "21")
c.metric("Positive class", f"{positive:,}")
d.metric("Positive class (%)", f"{positive / total * 100:.2f}%" if total else "N/A")

st.header("Diabetes class distribution")
class_df = pd.DataFrame({"Class":["Non-diabetes","Diabetes / Prediabetes"],"Records":[negative,positive]})
left,right = st.columns(2)
left.plotly_chart(px.pie(class_df,names="Class",values="Records",hole=.4),use_container_width=True)
right.plotly_chart(px.bar(class_df,x="Class",y="Records",color="Class",text="Records"),use_container_width=True)

st.header("Descriptive statistics")
numeric = ["BMI","MentHlth","PhysHlth","Age","Education","Income","GenHlth"]
st.dataframe(filtered[numeric].describe().T.round(2),use_container_width=True)

st.header("BMI analysis")
left,right=st.columns(2)
left.plotly_chart(px.histogram(filtered,x="BMI",nbins=30,title="BMI distribution"),use_container_width=True)
right.plotly_chart(px.box(filtered,x="Diabetes_binary",y="BMI",color="Diabetes_binary",title="BMI by target class"),use_container_width=True)

st.header("Age-wise positive-class percentage")
age_df=filtered.groupby("Age")["Diabetes_binary"].agg(["count","mean"]).reset_index()
age_df["Positive_Percentage"]=age_df["mean"]*100
age_labels={1:"18–24",2:"25–29",3:"30–34",4:"35–39",5:"40–44",6:"45–49",7:"50–54",8:"55–59",9:"60–64",10:"65–69",11:"70–74",12:"75–79",13:"80+"}
age_df["Age_Group"]=age_df["Age"].map(age_labels)
st.plotly_chart(px.bar(age_df,x="Age_Group",y="Positive_Percentage",text_auto=".2f"),use_container_width=True)

st.header("Health indicator analysis")
indicator=st.selectbox("Choose an indicator",["HighBP","HighChol","Smoker","PhysActivity","HeartDiseaseorAttack","Stroke","DiffWalk","HvyAlcoholConsump"])
indicator_df=filtered.groupby(indicator)["Diabetes_binary"].mean().mul(100).rename("Positive_Percentage").reset_index()
st.plotly_chart(px.bar(indicator_df,x=indicator,y="Positive_Percentage",color=indicator),use_container_width=True)

st.header("Correlation analysis")
corr_cols=["HighBP","HighChol","BMI","Smoker","Stroke","HeartDiseaseorAttack","PhysActivity","GenHlth","MentHlth","PhysHlth","DiffWalk","Age","Diabetes_binary"]
corr=filtered[corr_cols].corr()
st.plotly_chart(px.imshow(corr,text_auto=".2f",color_continuous_scale="RdBu_r",zmin=-1,zmax=1),use_container_width=True)
st.dataframe(corr["Diabetes_binary"].drop("Diabetes_binary").sort_values(ascending=False).to_frame("Correlation").round(3))
st.download_button("Download descriptive statistics",filtered.describe().T.to_csv(),"diabetes_statistics.csv","text/csv")
st.caption("Correlations show associations, not causation. This dashboard is for educational use.")
