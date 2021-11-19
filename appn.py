#import findspark
#import spark
import json
import time
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row,SQLContext
#from pyspark.sql.session import SparkSession

import sys
#import requests

#conf = SparkConf()
#spark = SparkSession(
#conf.setAppName("")
if __name__ == "__main__":
	sc= SparkContext(master="local[2]",appName="stream")
	ssc = StreamingContext(sc,10)
	lines= ssc.socketTextStream("localhost", 6100)
	sqlContext=SQLContext(sc)
	word=lines.flatMap(lambda line: line.split("\n"))
	#word=word.map(lambda lines: json.loads(lines))
	def cnf(rd):
		x=rd.take(1)
		print(x[0])
		y=json.loads(x[0])
		return y
		print(y)
	rdd=word.foreachRDD(cnf)
	#rdd.pprint()
	#rdd=word.map(lambda x: json.loads(x))	
	#r=json.loads(lines)
	print("new batch")
	ssc.start()
	ssc.awaitTermination()
	ssc.stop()
