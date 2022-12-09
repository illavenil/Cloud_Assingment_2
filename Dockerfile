from gcr.io/datamechanics/spark:platform-3.2.1-latest 
COPY predictionIlla.py ./
COPY Illa_model Illa_model
RUN pip3 install --upgrade pip --user
RUN pip3 install numpy pandas seaborn matplotlib Jinja2 pyspark==3.2.1 --user