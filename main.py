#Cameron Clapp
#Programming Assignment 1

from pyspark.ml.feature import Word2Vec
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.clustering import KMeans

import pandas as pd
from gensim import utils
import gensim.parsing.preprocessing as gsp
import boto3
from io import BytesIO

from collections import Counter

#Imports the object from S3 into the program
print("Reading File")
s3 = boto3.resource('s3')
obj = s3.Object(bucket_name='cs496hwdata', key='metadata.csv')
response = obj.get()
data = response['Body'].read()

# Only reads title and abstract since that's what we are interested in
print("Loading File into Pandas")
colums = ["title", "abstract"]
metaData = pd.read_csv(BytesIO(data),usecols=colums, nrows=100000)

# Drops all lines where abstract is null
metaData.dropna(subset=['abstract'], axis=0, inplace=True)

# Drops duplicate documents using title and abstract
metaData.drop_duplicates(
    subset=['title', 'abstract'], keep='last', inplace=True)

# Finds non english documents and fixes punctuation and combines for Doc2Vec
filters = [gsp.strip_tags,
           gsp.strip_punctuation,
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric,
           gsp.remove_stopwords,
           gsp.strip_short]

print("Cleaning Up Data")
#Steps for cleaning data
for index, row in metaData.iterrows():
    #Converts to string for later modules
    title = str(row[0])
    abstract = str(row[1])
    title = title.lower()
    abstarct = abstract.lower()
    title = title.replace("the", "")
    abstract = abstarct.replace("the", "")
    #Applys each gensim text cleanup on the text
    for f in filters:
        title = f(title)
        abstract = f(abstract)
    #Combines and saves the new text back to the data frame
    metaData.loc[index, 'title'] = index
    metaData.loc[index, 'abstract'] = title + abstract

#Connects to spark
spark = SparkSession.builder.appName("test").getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)

print("Creating Spark DF")
#Loads the pandas dataframe into spark format
sparkDataset = sqlContext.createDataFrame(metaData)
sparkDataset.repartition(50)


print("Training Word2Vec")
# Creates a Doc2Vec model and trains it
tokenizer = Tokenizer(inputCol="abstract", outputCol="tokens")
w2v = Word2Vec(vectorSize=300, minCount=0,
               inputCol="tokens", outputCol="features")
doc2vec_pipeline = Pipeline(stages=[tokenizer, w2v])
doc2vec_model = doc2vec_pipeline.fit(sparkDataset)
doc2vecs_df = doc2vec_model.transform(sparkDataset)

# Gets rid of extra columns for the processed dataset
finalDF = doc2vecs_df.drop("abstract", "tokens")

print("Running KMeans")
# Trains kmeans
kmeans = KMeans().setK(8).setSeed(1).setFeaturesCol("features")
model = kmeans.fit(finalDF)

#Joins the original rows with kmeans predictions
transformed = model.transform(finalDF).select('title', 'prediction')
rows = transformed.collect()
df_pred = sqlContext.createDataFrame(rows)

f = open("output.txt", "w")
#Collects dataframes for later use
collectedData = df_pred.collect()
collectedWords = doc2vecs_df.collect()
predic = []

#Appends each row into an easier to manage list
for row in collectedData:
    predic.append(row["prediction"])

#Count of each cluster
f.write("Clusters:\n")
count = Counter(predic)
print(count, file=f)
f.write("\n")


words = [[], [], [], [], [], [], [], []]
allWords = []
count = 0
#Finds top words in each cluster
for row in collectedData:
    words[row["prediction"]] += collectedWords[count]["tokens"]
    allWords += collectedWords[count]["tokens"]
    count += 1

for i in range(8):
    print("Cluster " + str(i+1) + " Top words\n", file=f)
    tempCount = Counter(words[i]).most_common(10)
    print(tempCount, file=f)
    f.write("\n")

topWordCount = Counter(allWords).most_common(3)
print("Top Word Count\n", file=f)
print(topWordCount, file=f)
f.write("\n")

f.close()

#Uploads the output to S3
s3 = boto3.resource('s3')
s3.Bucket('cs496hwdata').upload_file("output.txt", "output.txt")

sc.stop()
