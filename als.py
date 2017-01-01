import sys
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession

spark = SparkSession.builder.config(conf=SparkConf()).getOrCreate()

inputfile = str(sys.argv[1])
outputfile = str(sys.argv[2])
rank = int(sys.argv[3])

df = spark.read.csv(inputfile,header=True)
DF = df.select(df.userId.cast('int').alias('user'),df.movieId.cast('int').alias('item'),df.rating.cast('float')) \
	.cache()

from pyspark.ml.recommendation import ALS

als = ALS(rank=rank,maxIter=20,userCol='user',itemCol='item',ratingCol='rating')

spark.sparkContext.setCheckpointDir('checkpoint/')
ALS.checkpointInterval =2

model = als.fit(DF)

pred = model.transform(DF)

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(predictionCol = 'prediction',labelCol = 'rating',metricName = 'rmse')

pred.write.parquet(outputfile,'overwrite')
print 'rmse: '+ str(evaluator.evaluate(pred))
