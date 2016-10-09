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
    # config = EUTHC
    # config.valid_freq = 0.5
    # config.Model = AttentionEUTHM2
    # config.model_path = os.path.join(config.project_dir, "output/model/RAEUTH2/RAEUTH2.pkl")
    # entrance = AttentionEUTHE2(config)
    # entrance.train()

    #Single train AttentionEUTH
    # config = AttentionEUTHC
    # config.train_path = os.path.join(config.project_dir, "data/tweet/first_11_days.pkl")
    # config.model_path = os.path.join(config.project_dir, "output/sig vs com/RSAEUTH/RSAEUTH.pkl")
    # entrance = AttentionEUTHE(config)
    # entrance.train()

    #Compose train AttentionEUTH
    # config = ComAttentionEUTHC
    # config.train_path = os.path.join(config.project_dir, "data/tweet/first_11_days.pkl")
    # entrance = ComAttentionEUTHE(config)
    # entrance.train()

    #Single train EUTH
    # config = EUTHC
    # config.train_path = os.path.join(config.project_dir, "data/tweet/first_11_days.pkl")
    # config.model_path = os.path.join(config.project_dir, "output/sig vs com/RSEUTH/RSEUTH.pkl")
    # entrance = EUTHE(config)
    # entrance.train()

    #Compose train EUTH
    # config = ComEUTHC
    # config.train_path = os.path.join(config.project_dir, "data/tweet/first_11_days.pkl")
    # config.model_path = os.path.join(config.project_dir, "output/sig vs com/RCEUTH/RCEUTH.pkl")
    # entrance = ComEUTHE(config)
    # entrance.train()

    #Compose train ETH
    config = ComETHC
    config.T = 10
    config.train_path = os.path.join(config.project_dir, "data/tweet/first_11_days.pkl")
    config.model_path = os.path.join(config.project_dir, "output/sig vs com/RCETH/RCETH.pkl")
    entrance = ComETHE(config)
    entrance.train()


    #Compose train AEUTH on first 30 days
    # config = ComAttentionEUTHC
    # config.T = 30
    # config.neg_epoch = 20
    # config.train_path = os.path.join(config.project_dir, "data/tweet/first_31_days.pkl")
    # config.model_path = os.path.join(config.project_dir, "output/for first 31 days/RCAEUTH/RCAEUTH.pkl")
    # entrance = ComAttentionEUTHE(config)
    # entrance.train()

    #Compose train EUTH on first 30 days
    # config = ComEUTHC
    # config.T = 30
    # config.neg_epoch = 20
    # config.train_path = os.path.join(config.project_dir, "data/tweet/first_31_days.pkl")
    # config.model_path = os.path.join(config.project_dir, "output/for first 31 days/RCEUTH/RCEUTH.pkl")
    # entrance = ComEUTHE(config)
    # entrance.train()

    #Compose train ETH on first 30 days
    # config = ComETHC
    # config.T = 30
    # config.neg_epoch = 20
    # config.train_path = os.path.join(config.project_dir, "data/tweet/first_31_days.pkl")
    # config.model_path = os.path.join(config.project_dir, "output/for first 31 days/RCETH/RCETH.pkl")
    # entrance = ComETHE(config)
    # entrance.train()


    #Apply neg on first 10 days after full
    # config = NegEUTHC
    # config.valid_freq = 0.5
    # entrance = NegEUTHE(config)
    # entrance.train()

    #Apply neg on first 10 days after full
    # config = NegAttentionEUTHC
    # config.valid_freq = 0.5
    # entrance = NegAttentionEUTHE(config)
    # entrance.train()



    # entrance = FDUTHE()
    # entrance.do()
