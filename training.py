#!/usr/bin/env python
# coding: utf-8

# #### Illavenil Prabhakaran - Cloud

# In[1]:


from IPython.display import display
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #plotting
import seaborn as sns #good visualizing
import os
import warnings
warnings.filterwarnings('ignore')

from pyspark.context import SparkContext
from pyspark import SparkContext
sc = SparkContext('local')


# In[2]:


from pyspark.sql.session import SparkSession
spark = SparkSession.builder.appName('Illavenil').getOrCreate()


# In[3]:


path = "s3://illavenil/TrainingDataset.csv"
val_path = "s3://illavenil/ValidationDataset.csv"
df = spark.read.option("delimiter", ";").option("header", "true").option("inferSchema","true").csv(path)
val_df = spark.read.option("delimiter", ";").option("header", "true").option("inferSchema","true").csv(val_path)


# In[4]:


df.printSchema()


# In[5]:


from pyspark.sql.functions import isnan, when, count, col

df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()


# In[6]:


feature_col = df.drop('quality').columns
feature_col


# In[7]:


from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

# convert to vector column first
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=feature_col, outputCol=vector_col)
df_vector = assembler.transform(df).select(vector_col)

# get correlation matrix
matrix = Correlation.corr(df_vector, vector_col).collect()[0][0] 
corr_matrix = matrix.toArray().tolist() 
corr_matrix_df = pd.DataFrame(data=corr_matrix, columns = feature_col, index=feature_col) 
corr_matrix_df .style.background_gradient(cmap='coolwarm').set_precision(2)


# In[8]:


train_assembler = VectorAssembler(inputCols=feature_col,outputCol="features")
df_train = assembler.transform(df)
df_train.show()


# In[9]:


from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(labelCol = 'quality' , featuresCol = "corr_features")
model = dt.fit(df_train)


# In[10]:


model.transform(df_train)


# In[11]:


model.write().overwrite().save("s3://illavenil/Illa_model")


# In[12]:


model.load("s3://illavenil/Illa_model")


# In[13]:


display(model)


# In[14]:


validation_assembler = VectorAssembler(inputCols=feature_col,outputCol="features")
df_validation = assembler.transform(val_df)
df_validation.show()


# In[15]:


predictions = model.transform(df_validation)
predictions.select('quality', 'rawPrediction', 'prediction', 'probability').show()


# In[20]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %s" % (accuracy))
print("Test Error = %s" % (1.0 - accuracy))


# In[17]:


from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import FloatType
import pyspark.sql.functions as F

preds_and_labels = predictions.select(['prediction','quality']).withColumn('quality', F.col('quality').cast(FloatType())).orderBy('prediction')
preds_and_labels = preds_and_labels.select(['prediction','quality'])
metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
print(metrics.confusionMatrix().toArray())


# In[21]:


#f1_score
my_mc_lr = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='quality', metricName='f1')
print("F1 :", my_mc_lr.evaluate(predictions))

