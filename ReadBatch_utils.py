# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 20:17:22 2017

@author: Administrator
"""

import numpy as np
import nltk
import json
import codecs
import string
import re
from nltk.corpus import stopwords
import collections

class ReadBatch(object):
    def __init__(self, lenth_max ,model = "train", load_size=100, vocabulary_size=10000, class_num=5):
        self.load_size = load_size
        self.offset = 0
        self.epochs = 0
        self.lenth_max = lenth_max
        if model =="train":
            text_list,self.label_list = self.loaddata("E:\\NLP\\Yelp-test\\review_10001", load_size,class_num, model)
        else:
            text_list,self.label_list = self.loaddata("E:\\NLP\\Yelp-test\\review_10001", load_size,class_num, model)
        
        stop_words = stopwords.words('english')
        stop_punctuation = string.punctuation.replace('.','').replace('?',' ?').replace(' ','').replace('!','')
        
        ### text_list_normal:   shape----> [all_text,str(words)]
        text_list_normal= self.normalize(text_list, stop_words, stop_punctuation)
        
        word_diction, wordlist, text_list_normal = self.make_dictionary(text_list_normal, vocabulary_size=vocabulary_size)
        
        textnum = self.text2num(wordlist,word_diction,text_list_normal)
        
        self.text_num_normal = self.text_num2normal(textnum)
    
    def loaddata(self, path, load_size=100, class_num=5, model="train"):
        a = 0
        text_list=[]
        label_list = []
        if model =="train":
            for line in open(path,'rb'):
                if a == 0:
                    a +=1
                    continue
                elif a<load_size+1 :
                    obj = json.loads(line)
                    text_list.append(obj["text"])
                    label_list.append(obj["stars"])
                    a +=1
                else:
                    break
        else:
            for line in open(path,'rb'):
                if a < load_size:
                    a +=1
                    continue
                elif a < load_size*2 :
                    obj = json.loads(line)
                    text_list.append(obj["text"])
                    label_list.append(obj["stars"])
                    a +=1
                else:
                    break
        ## make the label to be "one-hot" arrary
        label = np.zeros([load_size, class_num])
        for i,x in enumerate(label_list):
            label[i][x-1]  = 1
        label = np.array(label)
        return text_list,label

    def normalize(self, text_list,stop_words, stop_punctuation):
        text_list = [text.lower() for text in text_list]
    
        text_list = [text.replace('\n','') for text in text_list]
#         text_list = [text.replace('.'," .").replace('?',' ?') for text in text_list]
        text_list = [''.join(x for x in text if x not in stop_punctuation) for text in text_list]
    
        text_list = [''.join(x for x in text if x not in "0123456789") for text in text_list]
#         text2word = [text.split() for text in text_list]
        ## text2word:size [text, words]
        text2word = [re.split(r'[! ?.]',text) for text in text_list]
        # text2word = [[sentence.split() for sentence in text] for text in text2sentence]
        # text2word = [sentence.split() for text in text2sentence for sentence in text ]
#         text_list = [' '.join(c for c in text if c not in stop_words)for text in text2word]
        text_list = [[' '.join(word for word in text if word not in stop_words)] for text in text2word]
        return text_list

    def make_dictionary(self, text_list_normal , vocabulary_size):
        ###    text2word:shape----->[all_text,words]
        text2word = [sentence.split() for text in text_list_normal for sentence in text]
        wordlist = [word for text in text2word for word in text]
        ## 返回前vocabulary_size个常见的词以及频数的tuple（word, num），且按顺序排列
        word_diction = collections.Counter(wordlist).most_common(vocabulary_size)
        wordlist = [word[0] for word in word_diction]
        word_diction = {word:index for index,word in enumerate(wordlist)}
        return word_diction,wordlist, text2word

    def text2num(self, wordlist,word_diction, text_list_normal):
        textnum = [[word_diction[word] for word in text if word in wordlist] for text in text_list_normal]
        ### delete the sentence whice is:[]
        for text in textnum:
            # for sentence in text:
            if text==[]:
                textnum.remove([])
        return textnum

    def text_num2normal(self, textnum):
        text_num_normal = []
        for text in textnum:
            if len(text) < self.lenth_max:
                zero = [0]*(self.lenth_max-len(text))
                text = text+zero
            else:
                text = text[0:self.lenth_max]
            text_num_normal.append(text)
        text_num_normal = np.array(text_num_normal)
        return text_num_normal
        
    def next_text(self, batch_size=10):
        start = self.offset
        self.offset += batch_size
        if self.offset > self.load_size:
            self.epochs += 1
            print("finish training"+str(self.epochs)+"times")
            num = np.arange(self.load_size)
            np.random.shuffle(num)
            self.text_num_normal = self.text_num_normal[num]
            self.label_list = self.label_list[num]
            self.offset = batch_size
            start = 0
        end = self.offset
        ## return self.text_num_normal[point] size: [text_size, sentence_size, word_size]
        ## self.label_list[point:point+1] : size:[text_size, num_class]
        return self.text_num_normal[start:end], self.label_list[start:end]
    
if __name__ == "__main__":
    readbatch = ReadBatch(lenth_max=100)
    for i in range(105):
        a,b=readbatch.next_text()
#    a,b = readbatch.next_text()
    print(readbatch.lenth_max)
    print(a.shape)
    print(a[0])
    print(type(b))