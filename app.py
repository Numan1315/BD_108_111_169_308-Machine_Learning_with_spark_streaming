import findspark
#import spark
import json
findspark.init()
import time
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row,SQLContext
from pyspark.sql.session import SparkSession

import sys
import requests

conf = SparkConf()
#spark = SparkSession(
#conf.setAppName("")
sc= SparkContext(conf = conf)
spark = SparkSession(sc)
ssc = StreamingContext(sc)
#ssc.checkpoint('checkpoint')
datastream = ssc.socketTextStream("localhost", 6100)
x = datastream.split('/n')
x.foreachRDD(
#tweet = datastream.map(lam)	
w = spark.read.json(sc.parallelize(datastream))

ssc.start()
ssc.awaitTermination()
