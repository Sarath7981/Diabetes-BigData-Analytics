
import streamlit as st
from src.prediction import predict_diabetes

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Diabetes Detection and Prediction")
st.subheader("Big Data Analytics and Machine Learning")

st.write(
    "Enter the health indicators below to generate "
    "a machine-learning-based prediction."
)

st.warning(
    "This application is for educational purposes only. "
    "It does not provide a medical diagnosis."
)

st.divider()

# Input form
with st.form("diabetes_form"):

    st.header("1. Health Information")

    col1, col2 = st.columns(2)

    with col1:
        high_bp = st.selectbox("High Blood Pressure", [0, 1],
                               format_func=lambda x: "Yes" if x else "No")

        high_chol = st.selectbox("High Cholesterol", [0, 1],
                                 format_func=lambda x: "Yes" if x else "No")

        chol_check = st.selectbox("Cholesterol Check in Last 5 Years", [0, 1],
                                  format_func=lambda x: "Yes" if x else "No")

        bmi = st.number_input("BMI", min_value=1, max_value=100, value=25)

        smoker = st.selectbox("Smoked at Least 100 Cigarettes", [0, 1],
                              format_func=lambda x: "Yes" if x else "No")

        stroke = st.selectbox("History of Stroke", [0, 1],
                              format_func=lambda x: "Yes" if x else "No")

        heart_disease = st.selectbox(
            "Heart Disease or Heart Attack",
            [0, 1],
            format_func=lambda x: "Yes" if x else "No"
        )

        phys_activity = st.selectbox(
            "Physical Activity",
            [0, 1],
            format_func=lambda x: "Yes" if x else "No"
        )

        fruits = st.selectbox("Consume Fruits Daily", [0, 1],
                              format_func=lambda x: "Yes" if x else "No")

        veggies = st.selectbox("Consume Vegetables Daily", [0, 1],
                               format_func=lambda x: "Yes" if x else "No")

        alcohol = st.selectbox(
            "Heavy Alcohol Consumption",
            [0, 1],
            format_func=lambda x: "Yes" if x else "No"
        )

    with col2:
        healthcare = st.selectbox("Have Healthcare Coverage", [0, 1],
                                  format_func=lambda x: "Yes" if x else "No")

        cost = st.selectbox(
            "Could Not Visit Doctor Due to Cost",
            [0, 1],
            format_func=lambda x: "Yes" if x else "No"
        )

        general_health = st.selectbox(
            "General Health",
            [1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "Excellent",
                2: "Very Good",
                3: "Good",
                4: "Fair",
                5: "Poor"
            }[x]
        )

        mental_health = st.slider(
            "Days of Poor Mental Health (Last 30 Days)",
            0, 30, 0
        )

        physical_health = st.slider(
            "Days of Poor Physical Health (Last 30 Days)",
            0, 30, 0
        )

        difficulty_walking = st.selectbox(
            "Difficulty Walking or Climbing Stairs",
            [0, 1],
            format_func=lambda x: "Yes" if x else "No"
        )

        sex = st.selectbox(
            "Sex (Dataset Category)",
            [0, 1],
            format_func=lambda x: {
                0: "Female",
                1: "Male"
            }[x]
        )

        age = st.selectbox(
            "Age Category",
            list(range(1, 14)),
            format_func=lambda x: {
                1: "18–24", 2: "25–29", 3: "30–34",
                4: "35–39", 5: "40–44", 6: "45–49",
                7: "50–54", 8: "55–59", 9: "60–64",
                10: "65–69", 11: "70–74",
                12: "75–79", 13: "80 or older"
            }[x]
        )

        education = st.selectbox(
            "Education Level",
            list(range(1, 7)),
            format_func=lambda x: {
                1: "Never attended / Kindergarten",
                2: "Grades 1–8",
                3: "Grades 9–11",
                4: "High school graduate",
                5: "Some college / Technical school",
                6: "College graduate"
            }[x]
        )

        income = st.selectbox(
            "Annual Household Income Category",
            list(range(1, 9)),
            format_func=lambda x: {
                1: "Below $10,000",
                2: "$10,000–$14,999",
                3: "$15,000–$19,999",
                4: "$20,000–$24,999",
                5: "$25,000–$34,999",
                6: "$35,000–$49,499",
                7: "$50,000–$74,999",
                8: "$75,000 or more"
            }[x]
        )

    st.divider()

    submitted = st.form_submit_button(
        "Generate Prediction",
        type="primary",
        use_container_width=True
    )

# Generate prediction
if submitted:

    user_data = [
        high_bp,
        high_chol,
        chol_check,
        bmi,
        smoker,
        stroke,
        heart_disease,
        phys_activity,
        fruits,
        veggies,
        alcohol,
        healthcare,
        cost,
        general_health,
        mental_health,
        physical_health,
        difficulty_walking,
        sex,
        age,
        education,
        income
    ]

    try:
        result = predict_diabetes(user_data)

        st.divider()
        st.header("2. Prediction Result")

        if result["prediction"] == 1:
            st.error("Model Prediction: Diabetes-positive class")
        else:
            st.success("Model Prediction: Diabetes-negative class")

        st.metric(
            "Estimated Positive-Class Probability",
            f'{result["probability"]}%'
        )

        st.progress(result["probability"] / 100)

        st.info(
            "This probability is produced by the trained model. "
            "It is not a calibrated medical risk estimate. "
            "Consult a qualified healthcare professional for "
            "testing and medical advice."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
