#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyspark 
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Wine Quality Prediction").enableHiveSupport().getOrCreate()


# In[2]:


df = spark.read.csv('TrainingDataset.csv',header='true', inferSchema='true', sep=';')


# In[3]:


new_column_name_list= list(map(lambda x: x.replace("\"\"", ""), df.columns))

df = df.toDF(*new_column_name_list)

df = df.withColumnRenamed("quality\"", "quality")


# In[4]:


def isTasty(quality):
    if quality >= 7:
        return 1
    else:
        return 0


# In[5]:


from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
tasty_udf_int = udf(isTasty, IntegerType())


# In[7]:


df_tasty = df.withColumn("tasty", tasty_udf_int('quality'))


# In[8]:


featureColumns = ["alcohol", "volatile acidity", "sulphates", "citric acid", "total sulfur dioxide", "density"]


# In[9]:


from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=featureColumns, 
                            outputCol="features")


# In[10]:


trainingData = assembler.transform(df_tasty).select('features', 'tasty')


# In[11]:


df_test = spark.read.csv('ValidationDataset.csv',header='true', inferSchema='true', sep=';')


# In[12]:


new_column_name_list= list(map(lambda x: x.replace("\"\"", ""), df.columns))

df_test = df_test.toDF(*new_column_name_list)

df_test = df_test.withColumnRenamed("quality\"", "quality")


# In[13]:


df_test_tasty = df_test.withColumn("tasty", tasty_udf_int('quality'))


# In[14]:


featureColumns_test = ["alcohol", "volatile acidity", "sulphates", "citric acid", "total sulfur dioxide", "density"]


# In[15]:


assembler_test = VectorAssembler(inputCols=featureColumns_test, 
                            outputCol="features")


# In[16]:


testData = assembler_test.transform(df_test_tasty).select('features', 'tasty')


# In[17]:


from pyspark.ml.classification import GBTClassifier


# In[18]:


gbt = GBTClassifier(maxIter=15).setLabelCol("tasty") 


# In[19]:


gbtModel = gbt.fit(trainingData)


# In[ ]:


gbt_preds = gbtModel.transform(testData)


# In[ ]:


gbt_evaluator = MulticlassClassificationEvaluator(
    labelCol='tasty', predictionCol="prediction", metricName="f1")
gbt_f1 = gbt_evaluator.evaluate(gbt_preds)
print("f-score on GBT = %g" % gbt_f1)


# In[ ]:


from pyspark.ml.classification import RandomForestClassifier

rfc = RandomForestClassifier(featuresCol='features',labelCol='tasty', numTrees=28)


# In[ ]:


rfc_model = rfc.fit(trainingData)


# In[ ]:


rfc_preds = rfc_model.transform(testData)


# In[ ]:


rfc_evaluator = MulticlassClassificationEvaluator(
    labelCol='tasty', predictionCol="prediction", metricName="f1")
rfc_f1 = rfc_evaluator.evaluate(rfc_preds)
print("f-score on RFC = %g" % rfc_f1)


# In[ ]:


from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'tasty', maxDepth =2)


# In[ ]:


dtModel = dt.fit(trainingData)


# In[ ]:


dt_preds = dtModel.transform(testData)


# In[ ]:


dt_evaluator = MulticlassClassificationEvaluator(
    labelCol='tasty', predictionCol="prediction", metricName="f1")
dt_f1 = dt_evaluator.evaluate(dt_preds)
print("f-score on DT = %g" % dt_f1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




