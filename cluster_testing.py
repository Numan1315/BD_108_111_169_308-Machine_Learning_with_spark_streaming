import nltk
from pyspark.sql.functions import monotonically_increasing_id, row_number
from pyspark.sql import Window
from nltk.stem import WordNetLemmatizer,PorterStemmer
import numpy as np
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.streaming import StreamingContext
import pyspark.sql.types as tp
#from pyspark.ml import Pipeline
from pyspark.sql import Row,SQLContext,SparkSession,functions as F
#from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.feature import StopWordsRemover, Word2Vec, Tokenizer
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import Row
from sparknlp.annotator import LemmatizerModel
from pyspark.ml.feature import HashingTF
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.decomposition import TruncatedSVD
import pickle
global zz,zo,fz,fo
zz=0
zo=0
fz=0
fo=0


global tc
tc=0

	
def test_clust(X,y):
	global zz,zo,fz,fo
	filename = 'cluster.sav'
	global tc
	rd=TruncatedSVD(2)
	pc=rd.fit_transform(X)
	clf=pickle.load(open(filename,'rb'))
	tc+=1
	y_pred = clf.predict(X)
	for i,j,k in zip(pc,y,y_pred):
		print(i[0],i[1],j,k)
		if(j==0 and k==0):
			zz+=1
		elif(j==0 and k==1):
			zo+=1
		elif(j==4 and k==0):
			fz+=1
		elif(j==4 and k==1):
			fo+=1
	print("zz %d zo %d fz %d fo %d"%(zz,zo,fz,fo))

if __name__ == "__main__":
	sc= SparkContext(master="local[2]",appName="test")
	ssc = StreamingContext(sc,5)
	spark = SparkSession(sc)
	lines= ssc.socketTextStream("localhost", 6100)
	sql=SQLContext(sc)
	lemmatizer=WordNetLemmatizer()
	porter=PorterStemmer()
	word=lines.flatMap(lambda line: line.split("\n"))
	def cnf(rd):
		f0=[]
		f1=[]
		f2=[]
		#print(rd)
		df= spark.read.json(rd)
		#df=sqlContext.createDataFrame(rd)
		f=df.collect()
		for i in f:
			for k in i:
				f0.append(k[0])
				f1.append(k[1].strip())
		if(len(f0)!=0 and len(f1)!=0):
			x=sql.createDataFrame(zip(f0,f1,f1),schema=['Sentiment','Tweet1','Tweet'])
			x=x.withColumn('Tweet',F.regexp_replace('Tweet',r'http\S+',''))
			x=x.withColumn('Tweet',F.regexp_replace('Tweet','@\w+',''))
			x=x.withColumn('Tweet',F.regexp_replace('Tweet','#',''))
			x=x.withColumn('Tweet',F.regexp_replace('Tweet','RT',''))
			x=x.withColumn('Tweet',F.regexp_replace('Tweet',':',' '))
			x=x.withColumn('Tweet',F.regexp_replace('Tweet',r'[^\w ]',' '))
			x=x.withColumn('Tweet',F.regexp_replace('Tweet',r'[\d]',''))
			x=x.withColumn('Tweet',F.regexp_replace('Tweet',r'[\d]',''))
			x=x.withColumn('Tweet',F.regexp_replace('Tweet',r'\b[a-zA-Z]\b',''))
			x=x.withColumn('Tweet',F.regexp_replace('Tweet',r'\b[a,p,t,A,P,T,R,S,s,n,N][m,M,H,h,t,T,D,d]\b',''))
			x=x.withColumn('Tweet',F.regexp_replace('Tweet',' +',' '))
			x=x.withColumn('Tweet',F.regexp_replace('Tweet','^\s+|\s+$',''))
			y=x.select('Tweet').collect()
			val2 = [ ele.__getattr__('Tweet') for ele in y]
			for i,j in zip(val2,range(len(val2))):
				words=nltk.word_tokenize(i)
				#lemma=' '.join([lemmatizer.lemmatize(w) for w in words])
				stem=' '.join([porter.stem(w) for w in words])
				y[j]=stem

			b = sql.createDataFrame([(l,) for l in y], ['Stem'])
			x = x.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))
			b = b.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))
			x = x.join(b, x.row_idx == b.row_idx).\
				     drop("row_idx")
			
			from sklearn.feature_extraction.text import HashingVectorizer
			vectorizer = HashingVectorizer(lowercase=True,stop_words={'english'},analyzer='word',alternate_sign=False)
			val=x.select('Stem').collect()
			Xtest=[ele.__getattr__('Stem') for ele in val]
			#vectorizer.partial_fit(Xtrain)
			Xtest_vect=vectorizer.transform(Xtest)
			
			val2=x.select('Sentiment').collect()
			ytest=[ele.__getattr__('Sentiment') for ele in val2]
			test_clust(Xtest_vect,ytest)
			print(tc)
	word.foreachRDD(cnf)
	#rdd.pprint()
	#rdd=word.map(lambda x: json.loads(x))	
	#r=json.loads(lines)
	ssc.start()
	ssc.awaitTermination()
	print('done')
	ssc.stop()

