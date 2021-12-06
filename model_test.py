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
from sklearn.metrics import classification_report,confusion_matrix,precision_score,recall_score,accuracy_score,f1_score
import pickle
global nbc,cnfnb
nbc=0
cnfnb=0
global pc,cnfp
cnfp=0
pc=0
global pacc,cnfpac
cnfpac=0
pacc=0
global sgdc,cnfsgd
cnfsgd=0
sgdc=0

def test_nb(X,y):
	filename='nb.sav'
	global nbc,cnfnb
	clf=pickle.load(open(filename,'rb'))
	if(nbc==0):
		cnfnb=confusion_matrix(y,clf.predict(X))
		pcnb=precision_score(y,clf.predict(X),average='macro')
		rcnb=recall_score(y,clf.predict(X),average='macro')
		f1nb=f1_score(y,clf.predict(X),average='macro')
		accnb=accuracy_score(y,clf.predict(X))
		nbc=1
	else:
		nbc+=1
		cnfnb+=confusion_matrix(y,clf.predict(X))
		pcnb=precision_score(y,clf.predict(X),average='macro')
		rcnb=recall_score(y,clf.predict(X),average='macro')
		f1nb=f1_score(y,clf.predict(X),average='macro')
		accnb=accuracy_score(y,clf.predict(X))
	print("nb",confusion_matrix(y,clf.predict(X)),pcnb,rcnb,f1nb,accnb)
def test_sgd(X,y):
	filename='sgd.sav'
	global sgdc,cnfsgd
	clf=pickle.load(open(filename,'rb'))
	if(sgdc==0):
		print(sgdc)
		cnfsgd=confusion_matrix(y,clf.predict(X))
		pcnb=precision_score(y,clf.predict(X),average='macro')
		rcnb=recall_score(y,clf.predict(X),average='macro')
		f1nb=f1_score(y,clf.predict(X),average='macro')
		accnb=accuracy_score(y,clf.predict(X))
		sgdc=1
	else:
		sgdc+=1
		cnfsgd+=confusion_matrix(y,clf.predict(X))
		pcnb=precision_score(y,clf.predict(X),average='macro')
		rcnb=recall_score(y,clf.predict(X),average='macro')
		f1nb=f1_score(y,clf.predict(X),average='macro')
		accnb=accuracy_score(y,clf.predict(X))
	#print(classification_report(y, clf.predict(X), output_dict=True))
	#print(ypred,nbc)
	print('sgd',confusion_matrix(y,clf.predict(X)),pcnb,rcnb,f1nb,accnb)
def test_percept(X,y):
	filename='p.sav'
	global pc,cnfp
	clf=pickle.load(open(filename,'rb'))
	if(pc==0):
		cnfp=confusion_matrix(y,clf.predict(X))
		pcnb=precision_score(y,clf.predict(X),average='macro')
		rcnb=recall_score(y,clf.predict(X),average='macro')
		f1nb=f1_score(y,clf.predict(X),average='macro')
		accnb=accuracy_score(y,clf.predict(X))
		pc=1
	else:
		pc+=1
		cnfp+=confusion_matrix(y,clf.predict(X))
		pcnb=precision_score(y,clf.predict(X),average='macro')
		rcnb=recall_score(y,clf.predict(X),average='macro')
		f1nb=f1_score(y,clf.predict(X),average='macro')
		accnb=accuracy_score(y,clf.predict(X))
	print('percept',confusion_matrix(y,clf.predict(X)),pcnb,rcnb,f1nb,accnb)
def test_pac(X,y):
	filename='pac.sav'
	global pacc,cnfpac
	clf=pickle.load(open(filename,'rb'))
	if(pacc==0):
		cnfpac=confusion_matrix(y,clf.predict(X))
		pcnb=precision_score(y,clf.predict(X),average='macro')
		rcnb=recall_score(y,clf.predict(X),average='macro')
		f1nb=f1_score(y,clf.predict(X),average='macro')
		accnb=accuracy_score(y,clf.predict(X))
		pacc=1
	else:
		pacc+=1
		cnfpac+=confusion_matrix(y,clf.predict(X))
		pcnb=precision_score(y,clf.predict(X),average='macro')
		rcnb=recall_score(y,clf.predict(X),average='macro')
		f1nb=f1_score(y,clf.predict(X),average='macro')
		accnb=accuracy_score(y,clf.predict(X))
	print('pac',confusion_matrix(y,clf.predict(X)),pcnb,rcnb,f1nb,accnb)
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
			Xtest=[ele.__getattr__('Stem') for ele in val]
			#vectorizer.partial_fit(Xtrain)
			Xtest_vect=vectorizer.transform(Xtest)
			
			val2=x.select('Sentiment').collect()
			ytest=[ele.__getattr__('Sentiment') for ele in val2]
			test_nb(Xtest_vect,ytest)
			test_percept(Xtest_vect,ytest)
			test_pac(Xtest_vect,ytest)
			test_sgd(Xtest_vect,ytest)
			print(sgdc)
			if(sgdc==80):
				precisionnb=cnfnb[0][0]/(cnfnb[0][0]+cnfnb[0][1])
				recallnb=cnfnb[0][0]/(cnfnb[0][0]+cnfnb[1][0])
				f_1nb=2*precisionnb*recallnb/(precisionnb+recallnb)
				accuracynb=(cnfnb[0][0]+cnfnb[1][1])/(cnfnb[0][0]+cnfnb[1][0]+cnfnb[1][1]+cnfnb[0][1])
				print("nblast",cnfnb,precisionnb,recallnb,f_1nb,accuracynb)
				
				precisionnb=cnfsgd[0][0]/(cnfsgd[0][0]+cnfsgd[0][1])
				recallnb=cnfsgd[0][0]/(cnfsgd[0][0]+cnfsgd[1][0])
				f_1nb=2*precisionnb*recallnb/(precisionnb+recallnb)
				accuracynb=(cnfsgd[0][0]+cnfsgd[1][1])/(cnfsgd[0][0]+cnfsgd[1][0]+cnfsgd[1][1]+cnfsgd[0][1])
				print("sgdlast",cnfsgd,precisionnb,recallnb,f_1nb,accuracynb)
				
				precisionnb=cnfp[0][0]/(cnfp[0][0]+cnfp[0][1])
				recallnb=cnfp[0][0]/(cnfp[0][0]+cnfp[1][0])
				f_1nb=2*precisionnb*recallnb/(precisionnb+recallnb)
				accuracynb=(cnfp[0][0]+cnfp[1][1])/(cnfp[0][0]+cnfp[1][0]+cnfp[1][1]+cnfp[0][1])
				print("perceptlast",cnfp,precisionnb,recallnb,f_1nb,accuracynb)
				
				precisionnb=cnfpac[0][0]/(cnfpac[0][0]+cnfpac[0][1])
				recallnb=cnfpac[0][0]/(cnfpac[0][0]+cnfpac[1][0])
				f_1nb=2*precisionnb*recallnb/(precisionnb+recallnb)
				accuracynb=(cnfpac[0][0]+cnfpac[1][1])/(cnfpac[0][0]+cnfpac[1][0]+cnfpac[1][1]+cnfpac[0][1])
				print("paclast",cnfpac,precisionnb,recallnb,f_1nb,accuracynb)
				return
	word.foreachRDD(cnf)
	ssc.start()
	ssc.awaitTermination()
	ssc.stop()
