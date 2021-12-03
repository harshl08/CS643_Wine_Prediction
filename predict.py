# Oliver Alvarado
# Dr. Manoop Talasila
# CS643-851
# Programming Assignment 2 -- Prediction Application
import random
import sys

from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import col, desc
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

###############################################################################
############################# INITIALIZE SPARK ################################
###############################################################################
# Create the spark session context.
spark = SparkSession.builder.appName("predict").getOrCreate()
spark.sparkContext.setLogLevel("Error")
print("########## SPARK VERSION:", spark.version)
print("########## SPARK CONTEXT:", spark.sparkContext)

###############################################################################
################################ READ DATA ####################################
###############################################################################
# Read data.
print("Reading data from {}...".format(sys.argv[1]))
testing = spark.read.format("csv").load(sys.argv[1], header=True, sep=";")

# Change column names and extract feature names.
testing = testing.toDF("fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol", "label")
testing.show(5)

# Ensure proper data types.
testing = testing \
        .withColumn("fixed_acidity", col("fixed_acidity").cast(DoubleType())) \
        .withColumn("volatile_acidity", col("volatile_acidity").cast(DoubleType())) \
        .withColumn("citric_acid", col("citric_acid").cast(DoubleType())) \
        .withColumn("residual_sugar", col("residual_sugar").cast(DoubleType())) \
        .withColumn("chlorides", col("chlorides").cast(DoubleType())) \
        .withColumn("free_sulfur_dioxide", col("free_sulfur_dioxide").cast(IntegerType())) \
        .withColumn("total_sulfur_dioxide", col("total_sulfur_dioxide").cast(IntegerType())) \
        .withColumn("density", col("density").cast(DoubleType())) \
        .withColumn("pH", col("pH").cast(DoubleType())) \
        .withColumn("sulphates", col("sulphates").cast(DoubleType())) \
        .withColumn("alcohol", col("alcohol").cast(DoubleType())) \
        .withColumn("label", col("label").cast(IntegerType()))

# Extract feature names. 
features = testing.columns
features = features[:-1]

# Convert the read data to the proper feature vector with predicted label format.
va = VectorAssembler(inputCols=features, outputCol="features")
va_df = va.transform(testing)
va_df = va_df.select(["features", "label"])
testing = va_df

###############################################################################
##################### LOAD MODEL AND MAKE PREDICTIONS #########################
###############################################################################
# Load the model. 
print("Loading model from {}...".format(sys.argv[2]))
trModel = MultilayerPerceptronClassificationModel.load(sys.argv[2])

# Make predictions. 
print("Making predictions...")
predictions = trModel.transform(testing)

# Evaluate the model. 
print("Evaluating the model...")
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % accuracy)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1 = evaluator.evaluate(predictions)
print("F1 = %g " % f1)
print("Model prediction finished... terminating.")