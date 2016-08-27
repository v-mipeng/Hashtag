'''
Pickle dataset
'''
import cPickle
import sys
import datetime
import os
sys.path.append("..")

class BUTHD(object):
    '''
       Basic dataset with user-text-time-hashtag information.

       load dataset --> parse string type date --> provide samples of given date --> map user, hashtag, word to id -->
       --> construct indexable dataset --> construct shuffled or sequencial fuel stream
       '''

    def __init__(self, data_path):
        # Dictionary
        self.index = 0
        self.data_path  = data_path

    def __iter__(self):
        return self

    def next(self):
        if self.index >3:
            raise StopIteration
        l = self.prepare(self.index)
        self.index += 1
        return l

    def prepare(self, index):
        '''
        Prepare dataset
        :param data_path:
        :return:
        '''
        print("Preparing dataset...")
        l = []
        if index == 0 or index == 2:
            with open(self.data_path, 'r') as f:
                for line in f:
                    array = line.split('\t')
                    l.append(array[index])
        elif index == 1:
            with open(self.data_path, 'r') as f:
                for line in f:
                    array = line.split('\t')
                    l.append(array[index].split(' '))
        else:
            with open(self.data_path, 'r') as f:
                for line in f:
                    array = line.strip().split('\t')
                    l.append(self._parse_date(array[index]))
        print("Done!")
        return l


    def _parse_date(self, str_date):
        '''
        Parse string format date.

        Reference: https://docs.python.org/2/library/time.html or http://strftime.org/
        :return: A datetime.date object
        '''
        return datetime.datetime.strptime(str_date, "%a %b %d %H:%M:%S +0000 %Y").date()

if __name__ == "__main__":
    from config.hashtag_config import UTHC
    config = UTHC()
    project_dir = config.project_dir
    print(project_dir)
    dataset = BUTHD(os.path.join(project_dir, "data/unit test/posts.uth"))
    with open(os.path.join(project_dir, "data/unit test/posts.pkl"), 'wb+') as f:
        for data in dataset:
            cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)


