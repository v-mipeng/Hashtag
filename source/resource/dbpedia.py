'''
Add Dbpedia dataset.
Provide redirect, dis-ambiguation services
'''
import re
import codecs

title2type = {}
redirects = {}
title2indegree = {}

class DBpedia:
    
    def __init__(self,dbpedia_type_path = None, dbpedia_red_path = None, dbpedia_indegree_path = None):
        if dbpedia_type_path is None:
            self.dbpedia_type_path = "./input/tables/dbpedia/type.txt"
        else:
            self.dbpedia_type_path = dbpedia_type_path
        if dbpedia_red_path is None:
            self.dbpedia_red_path = "./input/tables/dbpedia/redirect.txt"
        else:
            self.dbpedia_red_path = dbpedia_red_path
        if dbpedia_indegree_path is None:
            self.dbpedia_indegree_path = "./input/tables/dbpedia/indegree.txt"
        else:
            self.dbpedia_indegree_path = dbpedia_indegree_path

    def get_match_entities(self, mention):
        assert mention is not None
        reds = self.get_reds(mention)
        if reds is None:
            reds = []
        reds.append(mention.replace(" ","_"))
        reds = list(set(reds))
        types = self.get_types(reds)
        indegrees = self.get_indegrees(reds)
        types_copy = []
        indegrees_copy = []
        for (type, indegree) in zip(types,indegrees):
            if indegree != 0:
                types_copy.append(type)
                indegrees_copy.append(indegree)
        if len(types_copy) == 0:
            return None
        else:
            return zip(types_copy,indegrees_copy)

    def get_reds(self, mention):
        if len(redirects) == 0:
            self.__load_reds()
        assert mention is not None
        mention = mention.lower()
        if mention in redirects:
            return redirects[mention]
        else:
            return None

    def get_types(self, titles):
        if len(title2type) == 0:
            self.__load_types()
        assert titles is not None
        types = []
        for title in titles:
            if title in title2type:
                types.append(title2type[title])
            else:
                types.append("UNKNOWN")        
        return types

    def get_indegrees(self, titles):
        if len(title2indegree) == 0:
            if len(title2type) == 0:
                self.__load_types()
            self.__load_indegrees()
        assert titles is not None
        indegrees = []
        for title in titles:
            if title in title2indegree:
                indegrees.append(title2indegree[title])
            else:
                indegrees.append(0)        
        return indegrees

    def __load_reds(self):
        with codecs.open(self.dbpedia_red_path,"r", "utf-8") as f:
            for line in f:
                array = line.strip().split('\t')
                array[0] = array[0].replace('_',' ').lower()
                if array[0] in redirects:
                    redirects[array[0]].append(array[1])
                else:
                    redirects[array[0]] = [array[1]]

    def __load_indegrees(self):
        with codecs.open(self.dbpedia_indegree_path,"r", "utf-8") as f:
            for line in f:
                array = line.strip().split('\t')
                try:
                    if array[0] in title2type:
                        title2indegree[array[0]] = int(array[1])
                except:
                    print(line)

    def __load_types(self):
        with codecs.open(self.dbpedia_type_path,"r", "utf-8") as f:
            for line in f:
                array = line.strip().split('\t')
                title2type[array[0]] = array[1]   
