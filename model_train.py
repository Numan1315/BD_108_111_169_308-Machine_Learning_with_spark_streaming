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
#from sparknlp.annotator import LemmatizerModel
from pyspark.ml.feature import HashingTF
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron,PassiveAggressiveClassifier,SGDClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
import pickle
global nbc
nbc=0
global pc
pc=0
global pacc
pacc=0
global sgdc
sgdc=0

def naivebayes(X,y):
	filename='nb.sav'
	global nbc
	if(nbc==0):
		clf=MultinomialNB(alpha=0.1)
		nbc=1
	else:
		clf=pickle.load(open(filename,'rb'))
		nbc+=1
	clf=clf.partial_fit(X, y,classes=[0,4])
	pickle.dump(clf,open(filename,'wb'))

def sgd(X,y):
	filename='sgd.sav'
	global sgdc
	if(sgdc==0):
		clf=SGDClassifier(loss='modified_huber',penalty='l2')
		sgdc=1
	else:
		clf=pickle.load(open(filename,'rb'))
		sgdc+=1
	clf=clf.partial_fit(X, y,classes=[0,4])
	pickle.dump(clf,open(filename,'wb'))

def percept(X,y):
	filename='p.sav'
	global pc
	if(pc==0):
		p=Perceptron(alpha=0.0001)
		pc=1
	else:
		p=pickle.load(open(filename,'rb'))
		pc+=1
	p=p.partial_fit(X,y,classes=[0,4])
	pickle.dump(p,open(filename,'wb'))
def pac(X,y):
	filename='pac.sav'
	global pacc
	if(pacc==0):
		p=PassiveAggressiveClassifier(C=0.003,loss='squared_hinge')
		pacc=1
	else:
		p=pickle.load(open(filename,'rb'))
		pacc+=1
	p.partial_fit(X, y,classes=[0,4])
	pickle.dump(p,open(filename,'wb'))

if __name__ == "__main__":
	sc= SparkContext(master="local[*]",appName="trial")
	ssc = StreamingContext(sc,5)
	spark = SparkSession(sc)
	lines= ssc.socketTextStream("localhost", 6100)
	sql=SQLContext(sc)
	porter=PorterStemmer()
	word=lines.flatMap(lambda line: line.split("\n"))
	def cnf(rd):
		f0=[]
		f1=[]
		f2=[]
		df= spark.read.json(rd)
		f=df.collect()
		for i in f:
			for k in i:
				f0.append(k[0])
				f1.append(k[1].strip())
		if(len(f0)!=0 and len(f1)!=0):
			x=sql.createDataFrame(zip(f0,f1),schema=['Sentiment','Tweet'])
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
				     drop("row_idx","Tweet")
			
			from sklearn.feature_extraction.text import HashingVectorizer
			vectorizer = HashingVectorizer(lowercase=True,stop_words={'english'},analyzer='word',alternate_sign=False)
			val=x.select('Stem').collect()
			Xtrain=[ele.__getattr__('Stem') for ele in val]
			#vectorizer.partial_fit(Xtrain)
			Xtrain_vect=vectorizer.transform(Xtrain)
			
			val2=x.select('Sentiment').collect()
			ytrain=[ele.__getattr__('Sentiment') for ele in val2]
			naivebayes(Xtrain_vect,ytrain)
			sgd(Xtrain_vect,ytrain)
			percept(Xtrain_vect,ytrain)
			pac(Xtrain_vect,ytrain)
			print(pacc)	
	word.foreachRDD(cnf)
	ssc.start()
	ssc.awaitTermination()
	ssc.stop()


