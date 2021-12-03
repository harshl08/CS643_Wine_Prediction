FROM datamechanics/spark:3.1-latest

ENV PYSPARK_MAJOR_PYTHON_VERSION=3

WORKDIR /opt/wine-pred-app
RUN conda install numpy

COPY predict.py .
ADD ValidationDataset.csv .
ADD model ./model/
