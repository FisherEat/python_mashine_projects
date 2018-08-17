#!/usr/bin/env python
# encoding=utf-8

d = {'mike':90, 'bob':900}
print(d['mike'])

def my_abs(x):
    if x >= 0:
        return x
    else:
        return -x


print(my_abs(-90))

g = (x * x for x in range(10))

print(g)

print(type("123"))

class Student(object):
    pass

s = Student()
s.name = 'mike'
print (s.name)

def set_age(self, age):
    self.age = age

from types import MethodType

class Student(object):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return 'Student object (name=%s)' % self.name

s = Student('jiang')
print(s)