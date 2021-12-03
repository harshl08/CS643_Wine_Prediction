# Oliver Alvarado
# Dr. Manoop Talasila
# CS643-851
# Programming Assignment 2 -- Parallel Training Implementation
import random
import sys
import numpy as np 

from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import col, desc
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

###############################################################################
############################# INITIALIZE SPARK ################################
###############################################################################
# Create the spark session context.
spark = SparkSession.builder.appName("train").getOrCreate()
spark.sparkContext.setLogLevel("Error")
print("########## SPARK VERSION:", spark.version)
print("########## SPARK CONTEXT:", spark.sparkContext)

###############################################################################
################################ READ DATA ####################################
###############################################################################
# Read data from HDFS. 
print("Reading data from {}...".format(sys.argv[1]))
training = spark.read.format("csv").load(sys.argv[1], header=True, sep=";")

# Change column names and extract feature names.
training = training.toDF("fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol", "label")
training.show(5, False)

# Ensure proper data types.
training = training \
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
features = training.columns
features = features[:-1]

# Convert the read data to the proper feature vector with predicted label format.
va = VectorAssembler(inputCols=features, outputCol="features")
va_df = va.transform(training)
va_df = va_df.select(["features", "label"])
training = va_df

###############################################################################
######################### TRAIN AND SAVE THE MODEL ############################
###############################################################################
# Specify the layers.
layers = [11, 8, 8, 8, 8, 10]

# Declare the model.
tr = MultilayerPerceptronClassifier(maxIter=1000, layers=layers, blockSize=64, stepSize=0.03, solver='l-bfgs')

# Fit the model.
print("Training...")
trModel = tr.fit(training)

# Save the model. 
print("Saving file to {}...".format(sys.argv[2]))
trModel.write().overwrite().save(sys.argv[2])
print("Model saved... terminating.")
