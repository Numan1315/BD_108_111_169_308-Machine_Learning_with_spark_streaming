import numpy as np
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.streaming import StreamingContext
import pyspark.sql.types as tp
#from pyspark.ml import Pipeline
from pyspark.sql import Row,SQLContext,SparkSession
#from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.feature import StopWordsRemover, Word2Vec, Tokenizer
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import Row

if __name__ == "__main__":
	sc= SparkContext(master="local[2]",appName="trial")
	ssc = StreamingContext(sc,5)
	spark = SparkSession(sc)
	lines= ssc.socketTextStream("localhost", 6100)
	sql=SQLContext(sc)
	

	word=lines.flatMap(lambda line: line.split("\n"))
	#word=word.map(lambda lines: json.loads(lines))
	def cnf(rd):
		f0=[]
		f1=[]
		#print(rd)
		df= spark.read.json(rd)
		#df=sqlContext.createDataFrame(rd)
		f=df.collect()
		for i in f:
			for k in i:
				f0.append(k[0])
				f1.append(k[1])
		if(len(f0)!=0 and len(f1)!=0):
			x=sql.createDataFrame(zip(f0,f1),schema=['Sentiment','Tweet'])
			x.show()
			print(x)
			tokenizer = Tokenizer(inputCol = 'Tweet' , outputCol = 'Words')
			tokenizerTrain = tokenizer.transform(x)
			tokenizerTrain.show(truncate = False)
			print(tokenizerTrain)
			swr = StopWordsRemover(inputCol = tokenizer.getOutputCol(), outputCol = 'MeaningfulWords')
			SwrRemovedTrain = swr.transform(tokenizerTrain)
			SwrRemovedTrain.show(truncate = False)
			print(SwrRemovedTrain)
			
			
			
			
		#print(df.collect())
		#print(f0)
		#x=spark.read.json(rd)
		#sqlContext.implicits._rdd.toDf()
		#=x.rdd.map(lambda a: (lambda b: [b[0],b[1],b[2]]))
		#print(y.collect())
		
	rdd=word.foreachRDD(cnf)
	#rdd.pprint()
	#rdd=word.map(lambda x: json.loads(x))	
	#r=json.loads(lines)
	print("new batch")
	ssc.start()
	ssc.awaitTermination()
	ssc.stop()

