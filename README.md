# Cloud_Assingment_2
Cloud Programming Assignment 2

To run Apache Spark on Docker,
## Creating Spark Cluster EMR
Followed the steps provided to create cluster in AWS
I followed the guide provided by TA to create the EMR instance.
Then i followed the following guide to install jupyter on the cluyster.
https://github.com/PiercingDan/spark-Jupyter-AWS
the commands thaty i used:
[SparkMaster] wget https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86_64.sh
[SparkMaster] sh Anaconda2-4.2.0-Linux-x86_64.sh

I put the following commands in a shell script jupyter_setup.sh.

export spark_master_hostname=SparkMasterPublicDNS
export memory=1000M 

PYSPARK_DRIVER_PYTHON=jupyter PYSPARK_DRIVER_PYTHON_OPTS="notebook --no-browser --port=7777" pyspark --master spark://$spark_master_hostname:7077 --executor-memory $memory --driver-memory $memory

then to do port forwarding, i did the following:
[LocalComputer] ssh -i ~/path/to/AWSkeypair.pem -N -f -L localhost:7776:localhost:7777 ec2-user@SparkMasterPublicDNS

now i can run spark cluster by using my jupyter in local.
then i looked up into creating sparkcontext variable, sparksession and then i loadedd my data.
to add my dataset to the cluster, i used the scp command
then i tried various ML models from Spark MLlib. i did a bit of data exploration and then saved my best model.
I then downloaded the directory of my saved model. 

To create the docker image, I used data mechanics spark image.
https://hub.docker.com/r/datamechanics/spark
then i setup spark home, installed my dependancies and then copied my model directory and my prediction file and built the image
then I run the image using: 
docker run -v "C:\Users\illav\Documents\Cloud Assignment -2\ValidationDataset.csv":/opt/spark/work-dir/ValidationDataset.csv --rm -i -t illa python3 predictionIlla.py
