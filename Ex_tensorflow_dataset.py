# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 19:49:51 2021

@author: User
"""

import tensorflow as tf
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#---------------------------------------------------------------------------------------
        데이터 입력 파이프 라인.
        tf.data.Dataset.from_tensors() 또는 tf.data.Dataset.from_tensor_slices()를 사용.
        동일한 형태의 데이터를 입력했을때, from_tensors 는 그대로 받고 from_tensor_slices는 
#---------------------------------------------------------------------------------------
ds1 = tf.data.Dataset.from_tensor_slices([1,2,3,9,7,8])
print(ds1)
print(ds1.element_spec)

ds2 = tf.data.Dataset.from_tensors([1,2,3,9,8,7])
print(ds2)
print(ds2.element_spec)

ds3 = tf.data.Dataset.from_tensor_slices([[1,2,3,9,7,8],[4,5,6,12,11,10]])
print(ds3.element_spec)

ds4 = tf.data.Dataset.from_tensors([[1,2,3,9,7,8],[4,5,6,12,11,10]])
print(ds4.element_spec)

# for loop를 사용한 dataset 탐색
for d in ds1:
    print(d)
    print(d.numpy())

for d in ds2 :
    print(d)
    print(d.numpy())

for d in ds3:
    print(d)

for d in ds4:
    print(d)

# iterator를 사용한 dataset 탐색
iterator = iter(ds1)
d = next(iterator)
print(d)
d = next(iterator)
print(d)

# reduce를 사용해 dataset의 모든 요소를 통합해 단일 결과를 생성.
ds1_reduce = ds1.reduce(0, lambda s, v : s+v)
print(ds1_reduce)



#------------------------------------------------------------------
       데이터셋 구조
#------------------------------------------------------------------
tf.random.uniform([2,3]) # list means shape of Tensor [#row, #column]

ds1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4,10]))
print(ds1)

# ds2 : (1차원 Tensor, 2차원 Tensor) 의 구조를 가진다.
#       첫번째 요소의 길이와 두번째 요소의 행 수가 동일해야한다.

ds2 = tf.data.Dataset.from_tensor_slices((tf.random.uniform([4]),
                                         tf.random.uniform([4,100],maxval=100, dtype=tf.int32)))
print(ds2)
# Error : Dimension 5 and 4 are not compatible
tf.data.Dataset.from_tensor_slices((tf.random.uniform([5]),
                                         tf.random.uniform([4,100],maxval=100, dtype=tf.int32)))


it = iter(ds2)
d = next(it)
print(d)

ds3 = tf.data.Dataset.zip((ds1, ds2))
print(ds3)
print(ds3.element_spec)

# zip은 차원이 달라도 가능
d_temp = tf.data.Dataset.from_tensor_slices(tf.random.uniform([5,5]))
ds4 = tf.data.Dataset.zip((d_temp, ds2))
print(ds4.element_spec)


#------Sparse tensor
ds5 = tf.data.Dataset.from_tensors(tf.SparseTensor(indices = [[0,0],[1,2]], values=[1,2], dense_shape=[3,4]))
print(ds5.element_spec)

for d in ds5 :
    print(d)

ds5.element_spec.value_type


#------------------------------------------------------------------
       입력 데이터 읽기
#------------------------------------------------------------------

train, test = tf.keras.datasets.fashion_mnist.load_data()

img, labels = train
img = img/255

ds = tf.data.Dataset.from_tensor_slices((img, labels))
ds

it = iter(ds)
d1, d2 = next(it)
print(d1)
print(d2)


#------------------------------------------------------------------
       데이터셋 가져오기
       tf.keras.utils.get_file을 사용해 사이트에 올라와있는 dataset을
       다운받을 수 있다.
#------------------------------------------------------------------
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0) # 그냥 load하면 column 이름이 없고, 1행에 다른 정보가 들어있다.
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

print(train.head())
print(test.head())

# pop을 통해 dataframe에서 아예 삭제 가능.
train_y = train.pop('Species')
test_y = test.pop('Species')


#------------------------------------------------------------------
       예제...
#------------------------------------------------------------------

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

dftrain.describe()
dfeval.describe()

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

# Example - Categorical data
dftrain["class"].unique()
tf.feature_column.categorical_column_with_vocabulary_list("class", dftrain["class"].unique())

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# Creating the model
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

# Training the Model
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

print(result['accuracy'])
print(result)

pred_dicts = list(linear_est.predict(eval_input_fn))
print(pred_dicts)
print(pred_dicts[0]['probabilities'])

probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
probs.plot(kind='hist', bins=20, title='predicted probabilities')



#------ tf.data.Dataset 의 batch와 epochs 설정
temp_ds = tf.data.Dataset.from_tensor_slices((dict(dftrain), y_train))
temp_ds2 = temp_ds.batch(32) # batch size가 32
temp_ds3 = temp_ds.batch(32).repeat(10) # batch size가 32이고 epochs를 10으로 설정.


