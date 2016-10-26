import pandas as pd
import sklearn as sk
import ujson
import dill
from sklearn.preprocessing import OneHotEncoder
from sklearn import pipeline
from sklearn.linear_model import LogisticRegressionCV
import itertools
import numpy as np
from scipy.sparse import hstack

#Get the hero names
with open('heroes.json') as data_file:    
    heroes = ujson.load(data_file)

names = [item['localized_name'] for item in heroes['heroes']]
ids =   [item['id'] for item in heroes['heroes']]

heroes = zip(ids, names)

hero_names = list()
for i in range(0,114):
    if i in ids:
        hero_names.append(names[ids.index(i)])
    else:
        hero_names.append('')

heroVocab = dict()
heroComboVocab = dict()
for i in range(0, 114):
    heroVocab[str(i)] = i

def combos(x):
    return list(itertools.combinations(sorted(x),2))

comboVocab = combos([str(item) for item in range(0,114)])
    
class HeroTransformer(sk.base.BaseEstimator, sk.base.TransformerMixin):
    #Select the columns given
    def __init__(self):
        pass

    def fit(self, X, y):
        self.heroVectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False, vocabulary = heroVocab)
        self.combosVectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
        self.combosVectorizer.fit(list(X['rad_heroes'].apply(lambda x: list(itertools.combinations(sorted(x),2)))) + list(X['dire_heroes'].apply(lambda x: list(itertools.combinations(sorted(x),2)))))
        return self

    def transform(self, X):
        print(X['rad_heroes'])
        rad_heroes = self.heroVectorizer.transform(X['rad_heroes'])
        dire_heroes = self.heroVectorizer.transform(X['dire_heroes'])
        her = rad_heroes - dire_heroes
        
        rad_com = self.combosVectorizer.transform(list(X['rad_heroes'].apply(lambda x: list(itertools.combinations(sorted(x),2)))))
        dire_com = self.combosVectorizer.transform(list(X['dire_heroes'].apply(lambda x: list(itertools.combinations(sorted(x),2)))))
        com = rad_com - dire_com
        return hstack((her,com))
        
#Read the processed data into a dataframe
df = pd.DataFrame.from_csv('filtered_data2.csv', sep=',')
df = df.reset_index()
    
#Create a new column with just the heroes and lanes on the radiant side
df['rad_heroes'] = df['heroes'].apply(lambda x: [item.strip() for item in x[1:-1].split(',')][0:5])
df['rad_lanes'] = df['hero_lanes'].apply(lambda x: [item.strip() for item in x[1:-1].split(',')][0:5])

#Do the same for dire
df['dire_heroes'] = df['heroes'].apply(lambda x: [item.strip() for item in x[1:-1].split(',')][5:10])
df['dire_lanes'] = df['hero_lanes'].apply(lambda x: [item.strip() for item in x[1:-1].split(',')][5:10])

model_pipe = pipeline.Pipeline([
  ('hero_transformer', HeroTransformer()),
  ('reg', LogisticRegressionCV())])
    
model_pipe.fit(df,df['radiant_win'] == True)

model = model_pipe.named_steps['reg']
dill.dump(model, open( "model.p", "wb" ))

heroVectorizer = model_pipe.named_steps['hero_transformer'].heroVectorizer
dill.dump(heroVectorizer, open( "hero_vectorizer.p", "wb" ))

combosVectorizer = model_pipe.named_steps['hero_transformer'].combosVectorizer
dill.dump(combosVectorizer, open( "combos_vectorizer.p", "wb" ))