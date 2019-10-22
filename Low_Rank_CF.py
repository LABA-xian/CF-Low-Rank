# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 01:26:47 2019

@author: LABA
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, Dense, Flatten, dot, BatchNormalization, Dropout
from keras.models import Model, load_model
import numpy as np

def train_test_split_fuc(data):
        
    train, tt = train_test_split(data, test_size=0.3)
    
    vlidation, test = train_test_split(tt, test_size=0.5)

        
    return train, vlidation, test

ratings = pd.read_csv('ratings.csv')

label  = LabelEncoder()
ratings.userId = pd.DataFrame({'userId':ratings['userId'].values}).apply(label.fit_transform)
ratings.movieId = pd.DataFrame({'movieId':ratings['movieId'].values}).apply(label.fit_transform)


row = list(ratings['userId'].unique())
clo = list(ratings['movieId'].unique())

row = sorted(row)
clo = sorted(clo)

matrix = pd.pivot_table(data = ratings, values='rating', index='userId', columns='movieId')
matrix.fillna(0, inplace=True)

users = ratings.userId.unique()
movies = ratings.movieId.unique()

userid2idx = {o:i for i, o in enumerate(users)}
movieid2idx = {o:i for i, o in enumerate(movies)}

ratings['userId'] = ratings['userId'].apply(lambda x: userid2idx[x])
ratings['movieId'] = ratings['movieId'].apply(lambda x: movieid2idx[x])

train, val , test = train_test_split_fuc(ratings)

user_input = Input(shape=(1,), name='user_input')
user_embedding = Embedding(len(users), 100, name='user_embedding')(user_input)
user_vec = Flatten(name='user_vec')(user_embedding)
user_batch = BatchNormalization(name='user_batch')(user_vec)

movie_input = Input(shape=(1,), name='movies_input')
movie_embedding = Embedding(len(movies), 100, name='movie_embeddubg')(movie_input)
movie_vec = Flatten(name='movie_vec')(movie_embedding)
movie_batch = BatchNormalization(name='movie_batch')(movie_vec)

sim = dot([user_batch, movie_batch], name='sim', axes=1)

dense = Dense(100, activation='relu', name='dense1')(sim)
dense = BatchNormalization(name='bn1')(dense)
dense = Dropout(0.3)(dense)
dense = Dense(1, activation='relu')(dense)

model = Model([user_input, movie_input], dense)

model.compile(optimizer='Adam', loss='mse', metrics=['acc'])
model.summary()

#history = model.fit([train.userId, train.movieId], train.rating,
#                    validation_data=([val.userId, val.movieId], val.rating),
#                    batch_size=32,
#                    epochs=10,
#                    verbose=1)
#
#model.save('LR_CF.h5')

model = load_model('LR_CF.h5')

movie_data = np.array(list(set(ratings.movieId)))[0:4]
user_data = np.array(list(14 for i in range(len(movie_data))))


predictions = model.predict([user_data, movie_data])
predictions = np.array([a[0] for a in predictions])
recommended_movie_ids = (-predictions).argsort()[:]
print(recommended_movie_ids)
print(predictions[recommended_movie_ids])

