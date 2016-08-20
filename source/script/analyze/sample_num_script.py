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

to_label_id = {
u"music.music": 0,
u"broadcast.content": 0,
u"book.written_work": 0,
u"award.award": 0,
u"body.part": 0,
u"chemistry.chemistry": 0,
u"time.event": 0,
u"food.food": 0,
u"language.language": 0,
u"location.location": 1,
u"organization.organization": 2,
u"people.person": 3,
u"computer.software": 4,
u"commerce.consumer_product": 4,
u"commerce.electronics_product": 4,
}

id2label = {
    0:"other",
    1:"location",
    2:"organization",
    3:"person",
    4:"product"
    }

basedir = config.basedir
files = [file for file in listdir(basedir) if file.endswith(".txt") and not file.startswith("data info")]
label2num = {}
for file in files:
    if os.path.isfile(os.path.join(basedir, file)):
        with codecs.open(os.path.join(basedir, file),"r") as f:
            for line in f:
                line = line.strip()
                array = line.split('\t')
                try:
                    label = id2label[to_label_id[array[1]]]
                except:
                    continue
                if label not in label2num:
                    label2num[label] = 1
                else:
                    label2num[label] += 1
writer = open(os.path.join(basedir,"data info.txt"),"w+")
for item in label2num.items():
    writer.write("%s\t%s\n" %(item[0],item[1]))
writer.close()

