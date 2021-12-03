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
import pickle
global c
c=0

def naivebayes(X,y):
	filename='nb.sav'
	global c
	from sklearn.naive_bayes import MultinomialNB
	from sklearn.model_selection import train_test_split
	#x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
	if(c==0):
		clf=MultinomialNB()
		c=1
	else:
		clf=pickle.load(open(filename,'rb'))
		c+=1
	clf=clf.partial_fit(X, y,classes=[0,4])
	#print(clf.score(x_test,y_test),"pf\n")
	#clf1 = MultinomialNB()
	#clf1.fit(x_train, y_train)
	#print(clf1.score(x_test,y_test),"rf\n")
	pickle.dump(clf,open(filename,'wb'))

if __name__ == "__main__":
	sc= SparkContext(master="local[2]",appName="trial")
	ssc = StreamingContext(sc,5)
	spark = SparkSession(sc)
	lines= ssc.socketTextStream("localhost", 6100)
	sql=SQLContext(sc)
	lemmatizer=WordNetLemmatizer()
	porter=PorterStemmer()
	word=lines.flatMap(lambda line: line.split("\n"))
	#word=word.map(lambda lines: json.loads(lines))
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
			Xtrain=[ele.__getattr__('Stem') for ele in val]
			#vectorizer.partial_fit(Xtrain)
			Xtrain_vect=vectorizer.transform(Xtrain)
			
			val2=x.select('Sentiment').collect()
			ytrain=[ele.__getattr__('Sentiment') for ele in val2]
			clf=naivebayes(Xtrain_vect,ytrain)
			print(c)
		
	word.foreachRDD(cnf)
	#rdd.pprint()
	#rdd=word.map(lambda x: json.loads(x))	
	#r=json.loads(lines)
	ssc.start()
	ssc.awaitTermination()
	print('done')
	ssc.stop()

