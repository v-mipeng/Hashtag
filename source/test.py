import os
from entrance import *
from dataset import *
from config import *


if __name__ ==  "__main__":
    config = UTHC()
    dataset = RUTHD(config)
    raw_dataset = dataset.get_dataset(reference_date='FIRST_DAY', date_offset=config.time_window-1, duration=config.time_window)
    fields = zip(*raw_dataset)
    texts = fields[config.text_index]
    hashtags = fields[config.hashtag_index]
    with open(os.path.join(config.project_dir, 'data/tweet/first_10_days.txt'), 'w+') as writer:
        for text,hashtag in zip(texts,hashtags):
            writer.write("{0}\n{1}\n".format(text, hashtag))
