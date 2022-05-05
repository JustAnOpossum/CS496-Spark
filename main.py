import boto3
import pandas as pd
from langdetect import detect
from gensim import utils
import gensim.parsing.preprocessing as gsp

from pyspark.ml.feature import Word2Vec
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.clustering import KMeans

#Imports the object from S3 into the program
#s3 = boto3.resource('s3')
#obj = s3.Object(bucket_name='cs496hwdata', key='metadata.csv')
#response = obj.get()
#data = response['Body'].read()

#Only reads title and abstract since that's what we are interested in
colums = ["title", "abstract"]
metaData = pd.read_csv('test.csv', usecols=colums)

#Drops all lines where abstract is null
metaData.dropna(subset = ['abstract'],axis = 0, inplace = True)

#Drops duplicate documents using title and abstract
metaData.drop_duplicates(subset=['title','abstract'],keep ='last',inplace=True)

#Finds non english documents and fixes punctuation and combines for Doc2Vec
filters = [gsp.strip_tags, 
           gsp.strip_punctuation,
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric,
           gsp.remove_stopwords, 
           gsp.strip_short]

drop = []
for index, row in metaData.iterrows():
	title = row[0]
	abstract = row[1]
	if detect(title) != 'en':
		drop.append(index)
		continue
	title = title.lower()
	abstarct = abstract.lower()
	for f in filters:
		title = f(title)
		abstract = f(abstract)
	metaData.loc[index, 'title'] = index
	metaData.loc[index, 'abstract'] = title + abstract
metaData.drop(drop, inplace=True)

sc = SparkContext("local", "test")
sqlContext = SQLContext(sc)

sparkDataset = sqlContext.createDataFrame(metaData)

#Creates a Doc2Vec model and trains it
tokenizer = Tokenizer(inputCol="abstract", outputCol="tokens")
w2v = Word2Vec(vectorSize=300, minCount=0, inputCol="tokens", outputCol="features")
doc2vec_pipeline = Pipeline(stages=[tokenizer,w2v])
doc2vec_model = doc2vec_pipeline.fit(sparkDataset)
doc2vecs_df = doc2vec_model.transform(sparkDataset)

#Gets rid of extra columns for the processed dataset
finalDF = doc2vecs_df.drop("abstract", "tokens")

#Trains kmeans
kmeans = KMeans().setK(8).setSeed(1).setFeaturesCol("features")
model = kmeans.fit(finalDF)

transformed = model.transform(finalDF).select('title', 'prediction')
rows = transformed.collect()
df_pred = sqlContext.createDataFrame(rows)
df_pred = sqlContext.createDataFrame(rows)