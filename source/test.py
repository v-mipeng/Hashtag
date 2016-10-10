


# if __name__ ==  "__main__":
#     config = UTHC()
#     dataset = RUTHD(config)
#     raw_dataset = dataset.get_dataset(reference_date='FIRST_DAY', date_offset=config.time_window-1, duration=config.time_window)
#     fields = zip(*raw_dataset)
#     texts = fields[config.text_index]
#     hashtags = fields[config.hashtag_index]
#     with open(os.path.join(config.project_dir, 'data/tweet/first_10_days.txt'), 'w+') as writer:
#         for text,hashtag in zip(texts,hashtags):
#             writer.write("{0}\n{1}\n".format(text, hashtag))

import numpy as np

# true_index = []
# pred_dist = []
# with open('result','r') as f:
#     for line in f:
#         line = line.strip()
#         index, pred = line.split('\t')
#         true_index.append(int(index))
#         preds = pred.split(' ')
#         pred_dist.append([float(value) for value in preds])
# true_index = np.array(true_index,dtype='int32')
# pred_dist = np.array(pred_dist)
#
# true_hashtags = []
# pred_hashtags = []
#
# with open('hashtagGenerate','r') as f:
#     for line in f:
#         line = line.strip()
#         true_hashtag,pred_hashtag = line.split(':')
#         true_hashtags.append(true_hashtag.strip())
#         pred_hashtags.append(pred_hashtag.strip().split('\t'))
#
class Base(object):
    def __init__(self):
        pass
    
    def A(self):
        print("Base:A")
        
    def B(self):
        print("Base:B")
        
class SubOne(Base):
    def __init__(self):
        super(SubOne, self).__init__()
    
    def A(self):
        print("SubOne:A")

    def B(self):
        print("SubOne:B")


class SubTwo(Base):
    def __init__(self):
        super(SubTwo, self).__init__()


    def A(self):
        print("SubTwo:A")


    def B(self):
        print("SubTwo:B")

class SubThree(SubOne, SubTwo):
    def __init__(self):
        SubTwo.__init__(self)
        SubOne.__init__(self)

    def A(self):
        SubOne.A(self)

    def B(self):
        SubTwo.B(self)


c = SubThree()

c.A()
c.B()

