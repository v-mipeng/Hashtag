#region Backup
# import numpy as np
# from config import *
# from dataset import *
# config = UTHC
# dataset = RUTHD(config)
# dataset.prepare()
# raw_dataset = dataset.get_data()
#
# sparse_per = np.arange(0.005,0.05,0.005)
# #Sparse user
# unique_users, unique_user_counts = np.unique(users, return_counts=True)
# total_user_num = len(unique_users)
# sparse_user_per = []
# sparse_user_threshold = []
# for per in sparse_per:
#     threshold = get_sparse_threshold(unique_user_counts, per)
#     sparse_user_threshold.append(threshold)
#     sparse_user_num = (unique_user_counts <= threshold).sum()
#     sparse_user_per.append(1.0*sparse_user_num/total_user_num)
# #Sparse hashtag
# unique_hashtags, unique_hashtag_counts = np.unique(hashtags, return_counts=True)
# total_hashtag_num = len(unique_hashtags)
# sparse_hashtag_per = []
# sparse_hashtag_threshold = []
# for per in sparse_per:
#     threshold = get_sparse_threshold(unique_hashtag_counts, per)
#     sparse_hashtag_threshold.append(threshold)
#     sparse_hashtag_num = (unique_hashtag_counts <= threshold).sum()
#     sparse_hashtag_per.append(1.0*sparse_hashtag_num/total_hashtag_num)
#
# # Sparse word
# texts = np.array(texts)
# words = np.concatenate(texts)
# unique_words, unique_word_counts = np.unique(words, return_counts=True)
# total_word_num = len(unique_words)
# sparse_word_per = []
# sparse_word_threshold = []
# for per in sparse_per:
#     threshold = get_sparse_threshold(unique_word_counts, per)
#     sparse_word_threshold.append(threshold)
#     sparse_word_num = (unique_word_counts <= threshold).sum()
#     sparse_word_per.append(1.0 * sparse_word_num / total_word_num)
#
# # Samples contain @
# sample_num_with_at = 0 #63536 /262712
# for text in texts:
#     for word in text:
#         if word.startswith('@'):
#             sample_num_with_at += 1
#             break
#         else:
#             pass
#
# # Unique @ number
# at2freq = {} # len:23526 freq: 106563 unique_words: 159153
# for text in texts:
#     for word in text:
#         if word.startswith('@'):
#             if word not in at2freq:
#                 at2freq[word] = 1
#             else:
#                 at2freq[word] += 1
#
# # Samples contain #h
# sample_num_with_hat = 0  # 172780 /262712
# for text in texts:
#     for word in text:
#         if word.startswith('#') and len(word) > 1:
#             sample_num_with_hat += 1
#             break
#
# # Unique #h number
# hat2freq = {}  # len:51915 freq: 337920 unique_words: 159153
# for text in texts:
#     for word in text:
#         if word.startswith('#') and len(word) >1:
#             if word not in hat2freq:
#                 hat2freq[word] = 1
#             else:
#                 hat2freq[word] += 1
#
#
# # Unique authors, hashtags and words per day
# unique_authors = []
# unique_hashtags = []
# unique_words = []
# word_num_per_sample = []
# for i in range(dataset.date_span):
#     data_by_day = dataset.get_dataset(reference_date=dataset.FIRST_DAY,
#                                           date_offset=i,
#                                           duration=1)
#     fields = zip(*data_by_day)
#     users = fields[config.user_index]
#     texts = numpy.array(fields[config.text_index])
#     hashtags = fields[config.hashtag_index]
#     unique_authors.append(len(set(users)))
#     unique_hashtags.append(len(set(hashtags)))
#     texts = np.concatenate(texts, axis = 0)
#     unique_words.append(len(set(texts)))
#     word_num_per_sample.append(len(texts)*1.0/len(users))
#
#
#endregion


#region Statistic author, hahstag and word occurrence frequency
from config import *
from dataset import *
import cPickle
import numpy as np
from matplotlib import pyplot as plt

# Do remotely
config = EUTHC
config.train_path = os.path.join(BasicConfig.project_dir, "data/tweet/first_31_days.pkl")
raw_dataset = RUTHD(config)
raw_dataset.prepare()
dataset = EUTHD(config)
config.T = 30
data = raw_dataset.get_dataset(reference_date='FIRST_DAY', date_offset=config.T-1, duration=config.T)
dataset._update_before_transform(data)

with open('statistic_10.pkl', 'wb+') as f:
    cPickle.dump(dataset.user2freq, f)
    cPickle.dump(dataset.hashtag2freq, f)
    cPickle.dump(dataset.word2freq, f)

# Do locally
with open("statistic_30.pkl", 'rb') as f:
    user2freq = cPickle.load(f)
    hashtag2freq = cPickle.load(f)
    word2freq = cPickle.load(f)
    
# Plot user frequency
user_freq, user_freq_count = np.unique(user2freq.values(),return_counts=True)
user_freq_count_norm = user_freq_count*1.0/user_freq_count.sum()
plt.plot(user_freq[0:100], user_freq_count_norm[0:100], '--')
plt.xlabel('post number')
plt.ylabel('percent of author')
plt.title('Percent of Author by Post Number')
plt.show()
# Plot hashtag frequency
hashtag_freq, hashtag_freq_count = np.unique(hashtag2freq.values(), return_counts=True)
hashtag_freq_count_norm = hashtag_freq_count*1.0/hashtag_freq_count.sum()
plt.plot(hashtag_freq[0:100], hashtag_freq_count_norm[0:100], '--')
plt.xlabel('post number')
plt.ylabel('percent of hashtag')
plt.title('Percent of Hashtag by Post Number')
plt.show()
# Plot word frequency
word_freq, word_freq_count = np.unique(word2freq.values(), return_counts=True)
word_freq_count_norm = word_freq_count*1.0/word_freq_count.sum()
plt.plot(word_freq[0:100], word_freq_count_norm[0:100], '--')
plt.grid(True)
plt.xlabel('occurrence')
plt.ylabel('percent of word')
plt.title('Percent of Word by Occurrence')
plt.show()

#endregion

#region Turn words into lowercase
for file in files:
    with open(file, 'rb') as f:
        dataset = cPickle.load(f)
    for item in dataset:
        new_text = []
        for word in item[config.text_index]:
            if word.startswith('#') or word.startswith('@'):
                new_text.append(word)
            else:
                new_text.append(word.lower())
        item[config.text_index] = new_text
    with open(file, 'wb+') as f:
        cPickle.dump(dataset, f)
#endregion
