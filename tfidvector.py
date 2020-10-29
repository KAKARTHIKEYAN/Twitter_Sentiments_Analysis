import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
d=pd.read_csv('C:\\Users\\KARTHIK\Downloads\\Live-Twitter-sentiment-analysis-web-app-master\\twitter-sentiment-analysis-master\\Twitter-sentiment-analysis\\App.csv')
x = d.iloc[:,-2].values
tv = TfidfVectorizer(max_df=0.90, min_df=2, stop_words='english',ngram_range=(1, 2),max_features=6000)
x = tv.fit_transform(x.astype('U'))
pickle.dump(tv,open("tfid1.pickle","wb"))