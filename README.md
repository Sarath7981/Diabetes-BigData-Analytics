# Diabetes Detection and Prediction Using Big Data Analytics

A student project that explores diabetes-risk classification using the UCI CDC Diabetes Health Indicators dataset, PySpark-based data processing, machine-learning models, and a Streamlit dashboard.

## Project overview

The workflow currently developed for this project:

1. Download the CDC Diabetes Health Indicators dataset from the UCI Machine Learning Repository.
2. Clean the dataset, remove duplicate records, and save the processed data in Parquet format.
3. Train and compare Logistic Regression, Random Forest, and Gradient Boosting classifiers.
4. Select a model using the F1-score and use it to generate predictions.
5. Explore dataset statistics and model results through a Streamlit dashboard.

## Dataset

- **Source:** UCI Machine Learning Repository — CDC Diabetes Health Indicators
- **UCI dataset ID:** 891
- **Original records:** 253,680
- **Original columns:** 22 (21 input features and the `Diabetes_binary` target)
- **Cleaned records in the current run:** 229,474
- **Target:** `Diabetes_binary` (0 = no diabetes; 1 = diabetes or prediabetes as defined by the dataset)

The target is survey-defined and should not be interpreted as a clinical diagnosis.

## Model results

Results from the current local training run:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7178 | 0.3201 | 0.7518 | 0.4490 | 0.8075 |
| Random Forest | 0.7211 | 0.3242 | 0.7595 | 0.4544 | 0.8136 |
| Gradient Boosting | 0.7112 | 0.3175 | 0.7729 | 0.4501 | 0.8161 |

Random Forest had the highest F1-score in this run. Gradient Boosting had the highest recall and ROC-AUC. These results depend on the train/test split and local execution environment.

## Technology stack

- Python 3.11
- Apache Spark / PySpark
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Plotly and Matplotlib
- Joblib
- UCI ML Repository Python package

## Run locally

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download and process the dataset:

```powershell
python download_data.py
python bigdata_processing.py
```

Train the models:

```powershell
python src/model_training.py
```

Launch the dashboard:

```powershell
streamlit run app.py
```

Run commands from the project root. The scripts expect the folder structure and data paths used in this repository.

## Disclaimer

This project is for educational and academic purposes only. Its predictions are not medical advice, a diagnosis, or a substitute for professional healthcare assessment. The dataset contains survey-based indicators, and model performance—especially precision—has limitations.

## Project status

The project was developed and tested locally. The source files, trained model, and result artifacts still need to be added to this GitHub repository. Do not upload private data or unnecessary generated files.
