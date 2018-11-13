import pandas as pd
import numpy as np
import string
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

df = pd.read_csv("profiles.csv")

features_to_remove = ['pets','diet','education']
df.drop(labels=features_to_remove, axis=1, inplace=True)

drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
df['drink_mapping'] = df.drinks.map(drink_mapping)
#print(df[['drink_mapping','drinks']])
#print(df.smokes.value_counts())
#print(df.drugs.value_counts())
smokes_mapping = {"no":0,"sometimes":1,"when drinking":2,"trying to quit":3,"yes":4}
df['smokes_mapping']=df.smokes.map(smokes_mapping)

drugs_mapping = {"never":0,"sometimes":1,"often":2}
df['drugs_mapping'] = df.drugs.map(drugs_mapping)

sex_mapping = {"f":0,"m":1}
df['sex_mapping'] = df.sex.map(sex_mapping)

orientation_mapping = {"bisexual":0,"gay":1,"straight":2}
df['orientation_mapping'] = df.orientation.map(orientation_mapping)

status_mapping = {"single":0,"available":1,"married":2, "seeing someone":3, "unknown":4}
df['status_mapping'] = df.status.map(status_mapping)

df.fillna({'sex_mapping':0, 'orientation_mapping':0,'status_mapping':0,'age':0,'drugs_mapping':0 })

essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]


# Removing the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)

df['all_essays'] = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
df['essay_len'] = df['all_essays'].apply(lambda x: len(x))



def avg(u):
	a= u.split()
	b=0
	for i in a:
		b+=1
	return b

def frecuences(a):
	b=0
	b= a.count('me')+a.count('I')
	return b

df['avg_word_length'] = df.apply(lambda row: avg(row['all_essays'])/row['essay_len'], axis=1)
df['frecuences_word'] = df.all_essays.apply(lambda x: frecuences(x))


essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]


# Removing the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
df['all_essays'] = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
df['essay_len'] = df['all_essays'].apply(lambda x: len(x))
df['income2'] = df['income'].apply(lambda x: x * -1 if x<0 else x)

feature_data = df[['essay_len']]
y = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
y_scaled = min_max_scaler.fit_transform(y)


x_data = df[['age']]
x = x_data.values
min_max_scaler_x = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler_x.fit_transform(x)




plt.scatter(x_scaled,y_scaled,alpha=0.4)
plt.show()


regr = linear_model.LinearRegression()

model = regr.fit(x_scaled,y_scaled)
print(model.coef_[0])
print(model.intercept_)

y_predict = model.predict(x_scaled)

plt.plot(x_scaled,y_predict)
plt.show()

