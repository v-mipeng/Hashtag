'''
mask words Occur less than 
'''
import config
import importlib
import os
import sys
import re

import codecs

from os import listdir

import numpy as np
import cPickle

with open('train3.pkl', 'rb') as f:
    cPickle.load(f)
    texts = cPickle.load(f)

unique_words = set()
for text in texts:
    unique_words = unique_words or set(text)
print(len(unique_words))