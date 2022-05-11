# CS496-Spark

# Problems
I was running out of time so I was only able to run the program on 100,000 rows of data instead of the full set.
 
# How to Run
1. Install conda, guide here https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
2. Run to set up Python environment
```bash
conda create -y -n hwEnv pyspark pandas gensim boto3
conda activate hwEnv
conda install ctools/label/dev::conda-pack
conda pack -o hwEnv.tar.gz
conda deactivate
```
3. Create an S3 bucket and edit main.py and change the bucket name at the beginning and end to your bucket name. Make sure you have a cluster with 4 EC2 m5.2xlarge instances
4. Submit Program to the Spark cluster, repalce envName with where your conda environment is
```bash
PYSPARK_PYTHON=/home/hadoop/hwEnv/bin/python \
spark-submit --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./environment/bin/python --conf "spark.yarn.am.waitTime=600000" --conf spark.rpc.message.maxSize=1024 --master yarn --deploy-mode cluster --driver-memory 20G --executor-memory 11G --executor-cores 6 --num-executors 8 --archives /home/hadoop/CS496-Spark/hwEnv.tar.gz#environment main.py
```

# Sources
https://www.adamsmith.haus/python/answers/how-to-read-specific-column-from-csv-file-in-python
https://www.kaggle.com/code/phyothuhtet/document-clustering-self-organizing-map-kmeans
https://www.kaggle.com/code/maksimeren/covid-19-literature-clustering
https://www.kaggle.com/code/islameladwiy/covid
https://rsandstroem.github.io/sparkkmeans.html
https://pub.towardsai.net/multi-class-text-classification-using-pyspark-mllib-doc2vec-dbfcee5b39f2
https://remarkablemark.org/blog/2020/08/26/python-iterate-csv-rows/
https://pypi.org/project/langdetect/
https://www.adamsmith.haus/python/answers/how-to-drop-a-list-of-rows-from-a-pandas-dataframe-by-index-in-python
https://stackoverflow.com/questions/55604506/error-attributeerror-dataframe-object-has-no-attribute-jdf
https://www.delftstack.com/howto/python/python-output-to-file/