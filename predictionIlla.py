#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
os.environ['SPARK_HOME']="../"

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import warnings
warnings.filterwarnings('ignore')

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark import SparkContext
sc = SparkContext('local')


# In[4]:


spark = SparkSession.builder.appName('IllaCloud').getOrCreate()

val_path = "ValidationDataset.csv"
val_data = spark.read.option("delimiter", ";").option("header", "true").option("inferSchema","true").csv(val_path)


feature_list = val_data.drop('quality').columns


from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
validation_assembler = VectorAssembler(inputCols=feature_list,outputCol="corr_features")
df_validation = validation_assembler.transform(val_data)
# df_validation.show()




from pyspark.ml.classification import DecisionTreeClassificationModel

dt = DecisionTreeClassificationModel.load("./Illa_model")



predictions = dt.transform(df_validation)




from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction",metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %s" % (accuracy))
print("Test Error = %s" % (1.0 - accuracy))


predictions.select('quality', 'prediction', 'probability').show()


# In[ ]:




