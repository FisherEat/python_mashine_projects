import pandas as pd

import numpy as np

import pandas.io

import pandas.io.parsers

arr1 = np.arange(10)
type(arr1)

s1 = pd.Series(arr1)
print(s1)


dic1 = {'a':10,'b':20,'c':30,'d':40,'e':50}
dic1
type(dic1)
s2 = pd.Series(dic1)
s2
type(s2)

print(s2)


arr2 = np.array(np.arange(12)).reshape(4,3)

df1 = pd.DataFrame(arr2)
print(df1)

s5 = pd.Series(np.array([10,15,20,30,55,80]),index = ['a','b','c','d','e','f'])
print(s5)


student = pd.io.parsers.read_csv('/Users/gl/Desktop/test_pandas.csv')

print(student)