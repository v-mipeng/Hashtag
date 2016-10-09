import numpy as np
from config import *
from dataset import *
config = UTHC
dataset = RUTHD(config)
dataset.prepare()
raw_dataset = dataset.get_data()

sparse_per = np.arange(0.005,0.05,0.005)
#Sparse user
unique_users, unique_user_counts = np.unique(users, return_counts=True)
total_user_num = len(unique_users)
sparse_user_per = []
sparse_user_threshold = []
for per in sparse_per:
    threshold = get_sparse_threshold(unique_user_counts, per)
    sparse_user_threshold.append(threshold)
    sparse_user_num = (unique_user_counts <= threshold).sum()
    sparse_user_per.append(1.0*sparse_user_num/total_user_num)
#Sparse hashtag
unique_hashtags, unique_hashtag_counts = np.unique(hashtags, return_counts=True)
total_hashtag_num = len(unique_hashtags)
sparse_hashtag_per = []
sparse_hashtag_threshold = []
for per in sparse_per:
    threshold = get_sparse_threshold(unique_hashtag_counts, per)
    sparse_hashtag_threshold.append(threshold)
    sparse_hashtag_num = (unique_hashtag_counts <= threshold).sum()
    sparse_hashtag_per.append(1.0*sparse_hashtag_num/total_hashtag_num)

# Sparse word
texts = np.array(texts)
words = np.concatenate(texts)
unique_words, unique_word_counts = np.unique(words, return_counts=True)
total_word_num = len(unique_words)
sparse_word_per = []
sparse_word_threshold = []
for per in sparse_per:
    threshold = get_sparse_threshold(unique_word_counts, per)
    sparse_word_threshold.append(threshold)
    sparse_word_num = (unique_word_counts <= threshold).sum()
    sparse_word_per.append(1.0 * sparse_word_num / total_word_num)

# Samples contain @
sample_num_with_at = 0 #63536 /262712
for text in texts:
    for word in text:
        if word.startswith('@'):
            sample_num_with_at += 1
            break
        else:
            pass

# Unique @ number
at2freq = {} # len:23526 freq: 106563 unique_words: 159153
for text in texts:
    for word in text:
        if word.startswith('@'):
            if word not in at2freq:
                at2freq[word] = 1
            else:
                at2freq[word] += 1

# Samples contain #h
sample_num_with_hat = 0  # 172780 /262712
for text in texts:
    for word in text:
        if word.startswith('#') and len(word) > 1:
            sample_num_with_hat += 1
            break

# Unique #h number
hat2freq = {}  # len:51915 freq: 337920 unique_words: 159153
for text in texts:
    for word in text:
        if word.startswith('#') and len(word) >1:
            if word not in hat2freq:
                hat2freq[word] = 1
            else:
                hat2freq[word] += 1


# Unique authors, hashtags and words per day
unique_authors = []
unique_hashtags = []
unique_words = []
word_num_per_sample = []
for i in range(dataset.date_span):
    data_by_day = dataset.get_dataset(reference_date=dataset.FIRST_DAY,
                                          date_offset=i,
                                          duration=1)
    fields = zip(*data_by_day)
    users = fields[config.user_index]
    texts = numpy.array(fields[config.text_index])
    hashtags = fields[config.hashtag_index]
    unique_authors.append(len(set(users)))
    unique_hashtags.append(len(set(hashtags)))
    texts = np.concatenate(texts, axis = 0)
    unique_words.append(len(set(texts)))
    word_num_per_sample.append(len(texts)*1.0/len(users))



