import os
import cPickle
from entrance import *
from config import *
from dataset import *


def tmp():
    config = UTHC
    raw_dataset = RUTHD(config)
    train_raw_dataset = raw_dataset.get_dataset(reference_date=raw_dataset.FIRST_DAY,
                                                     date_offset=config.T - 1,
                                                     duration=config.time_window)
    test_raw_dataset = raw_dataset.get_dataset(reference_date=raw_dataset.FIRST_DAY,
                                                    date_offset=config.T,
                                                    duration=1)
    with open(os.path.join(config.project_dir, "data/tweet/first_10_days.pkl"), "wb+") as f:
        cPickle.dump(train_raw_dataset, f)
    with open(os.path.join(config.project_dir, "data/tweet/eleventh_day.pkl"), "wb+") as f:
        cPickle.dump(test_raw_dataset, f)
    both_raw_dataset = raw_dataset.get_dataset(reference_date=raw_dataset.FIRST_DAY,
                                                    date_offset=config.T,
                                                    duration=config.time_window + 1)
    with open(os.path.join(config.project_dir, "data/tweet/first_11_days.pkl"), "wb+") as f:
        cPickle.dump(both_raw_dataset, f)


if __name__ ==  "__main__":
    config = ComETHC
    config.T = 10
    config.train_path = os.path.join(config.project_dir, "data/tweet/first_11_days.pkl")
    config.model_path = os.path.join(config.project_dir, "output/sig vs com/RCETH/RCETH.pkl")
    entrance = BaseEntrance(config)
    entrance.train()
