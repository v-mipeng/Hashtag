'''
Compare the result of two systems on the same test dataset
'''

import os
import ntpath

result_file_one = r"D:\Codes\Project\EntityTyping\Neural Entity Typing\slides\Multi time LSTM\train on satori bbn and conll\initialize emb with google word2vec\multi_time_lstm test on labelled samples.txt"
result_file_two = r"D:\Codes\Project\EntityTyping\Neural Entity Typing\slides\Multi time LSTM\train on satori bbn and conll\use dbpedia\multi_time_lstm test on labelled samples.txt"
comp_result_file = r"D:\Codes\Project\EntityTyping\Neural Entity Typing\slides\Multi time LSTM\train on satori bbn and conll\without vs with dbpedia.txt"

dir = ntpath.dirname(comp_result_file)
writer = open(comp_result_file, "w+")

pos_pos_writer = open(os.path.join(dir,"pos_pos.txt"),"w+")
pos_neg_writer = open(os.path.join(dir,"pos_neg.txt"),"w+")
neg_pos_writer = open(os.path.join(dir,"neg_pos.txt"),"w+")
neg_neg_writer = open(os.path.join(dir,"neg_neg.txt"),"w+")

pos_pos = 0
pos_neg = 0
neg_pos = 0
neg_neg = 0

result_ones = open(result_file_one, "r").readlines()
result_twos = open(result_file_two, "r").readlines()

if len(result_ones) != len(result_twos):
    raise Exception("The length of the results does not match!")



for result_one,result_two in zip(result_ones,result_twos):
    pos_one = True
    pos_two = True
    array_one = result_one.strip().split('\t')
    array_two = result_two.strip().split('\t')
    try:
        # The following 4 lines code is file format specific
        if array_one[1] != array_one[2]:    
            pos_one = False
        if array_two[1] != array_two[2]:
            pos_two = False
        if pos_one and pos_two:
            pos_pos_writer.write(result_one)
            pos_pos += 1
        if pos_one and not pos_two:
            pos_neg_writer.write(result_two)
            pos_neg += 1
        if not pos_one and pos_two:
            neg_pos_writer.write(result_one)
            neg_pos += 1
        if not pos_one and not pos_two:
            neg_neg_writer.write("%s\t%s\t%s\t%s\t%s\n" %(array_one[0],array_one[1],array_one[2], array_two[2],array_one[3]))
            neg_neg += 1
    except Exception as e:
        print(e.message)
        continue
        
pos_pos_writer.close()
pos_neg_writer.close()
neg_pos_writer.close()
neg_neg_writer.close()
writer.write("One | Two\tpos\tneg\n")
writer.write("pos      \t%s\t%s\n" %(pos_pos,pos_neg))
writer.write("neg      \t%s\t%s\n" %(neg_pos,neg_neg))
writer.close()

