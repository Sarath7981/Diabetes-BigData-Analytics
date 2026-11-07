
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, when, avg

# Create Spark session
spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("DiabetesBigDataProcessing")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Load dataset
input_path = "data/raw/diabetes.csv"
df = spark.read.csv(input_path, header=True, inferSchema=True)

print("\n========== ORIGINAL DATASET ==========")
print("Total records:", df.count())
print("Total columns:", len(df.columns))

# Display schema
df.printSchema()

# Check missing values
print("\n========== MISSING VALUES ==========")
missing = df.select([
    sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in df.columns
])
missing.show(truncate=False)

# Remove duplicate records
df_clean = df.dropDuplicates()

# Remove rows with missing values
df_clean = df_clean.dropna()

# Ensure target contains valid classes
df_clean = df_clean.filter(
    col("Diabetes_binary").isin(0, 1)
)

print("\n========== CLEANED DATASET ==========")
print("Records after cleaning:", df_clean.count())

# Target distribution
print("\n========== TARGET DISTRIBUTION ==========")
df_clean.groupBy("Diabetes_binary").count().show()

# Average BMI
print("\n========== AVERAGE BMI ==========")
df_clean.select(avg("BMI").alias("Average_BMI")).show()

# Save processed dataset
output_path = "data/processed/diabetes_parquet"

# Save using Pandas and PyArrow to avoid Windows Hadoop utilities
pdf = df_clean.toPandas()
pdf.to_parquet(output_path + ".parquet", index=False)

print("\nProcessed dataset saved successfully!")
print("Location:", output_path)

spark.stop()
