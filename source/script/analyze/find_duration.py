'''
Find the most approariate date duration of the training dataset. The following aspects are considered:

1. The number of items
2. The hashtag distribution in training dataset and test datasets
'''
import sys
sys.path.extend("../..")
import os
import cPickle
from collections import OrderedDict
import datetime

import numpy as np

from config import UTHC
from dataset import  RUTHD

def get_distribution(tags):
    assert tags is not None
    return dict(zip(*np.unique(tags, return_counts=True)))

def get_kl_dist(p_dist, q_dist):
    common_keys = set(p_dist.keys()) & set(q_dist.keys())
    p_scale = 0.
    q_scale = 0.
    for key in common_keys:
        p_scale += p_dist[key]
        q_scale += q_dist[key]
    kl_dist = 0.
    for key in common_keys:
        kl_dist += p_dist[key]/p_scale*np.log(p_dist[key]*q_scale/q_dist[key]/p_scale)
    return kl_dist

def get_coverage(train_dist, test_dist):
    common_keys = set(train_dist.keys()) & set(test_dist.keys())
    test_total = sum(test_dist.values())
    com_total = sum([test_dist[key] for key in common_keys])
    return 1.0*com_total/test_total

def get_kl_distances(config):
    dataset = RUTHD(config)
    dataset.prepare()
    fields = zip(*dataset.raw_dataset)
    hashtags = np.array(fields[config.hashtag_index])
    dates = np.array(fields[config.date_index])
    first_day = dataset.first_date
    last_day = dataset.last_date
    date_span = (last_day-first_day).days + 1
    del fields, dataset
    KLs = OrderedDict()
    AKLs = OrderedDict()
    CRs = OrderedDict()
    ACRs = OrderedDict()
    max_duration = min(date_span, 30)
    max_date_span = min(date_span, 60)
    for duration in range(1, max_duration):
        if duration == 10:
            pass
        kls = []
        kl = 0.
        crs = []
        cr = 0.
        for date_offset in range(duration, max_date_span):
            date_end = first_day + datetime.timedelta(days=date_offset -1)
            date_begin = date_end - datetime.timedelta(days = duration)
            idxes = np.logical_and(dates > date_begin , dates <= date_end)
            train_hashtags = hashtags[idxes]
            train_distribution = get_distribution(train_hashtags)
            date_begin = date_end+datetime.timedelta(days = 1)
            idxes = (dates == date_begin)
            test_hashtags = hashtags[idxes]
            test_distribution = get_distribution(test_hashtags)
            tmp = get_kl_dist(train_distribution, test_distribution)
            kls.append(tmp)
            kl += tmp
            tmp = get_coverage(train_distribution, test_distribution)
            crs.append(tmp)
            cr += tmp
        kl /= max_date_span-duration
        cr /= max_date_span-duration
        KLs[duration] = kls
        CRs[duration] = crs
        AKLs[duration] = kl
        ACRs[duration] = cr
    return AKLs, KLs, CRs, ACRs


if __name__ == "__main__":
    config = UTHC()
    print(config.project_dir)
    config.train_path = os.path.join(config.project_dir, 'data/tweet/first_11_days.pkl')
    AKLs, KLs, CRs, ACRs = get_kl_distances(config)
    akls = np.array(AKLs.values())
    best_duration = akls.argmin()
    result_path = os.path.join(config.project_dir, 'output/analysis/kl.txt')
    with open(result_path, "w+") as writer:
        writer.write("Average KL\n")
        writer.write("{0:10}{1:5}\n".format("duration","AKL"))
        for duration, akl in AKLs.iteritems():
            writer.write("{0:2d}\t{1:.4f}\n".format(duration,akl))

        writer.write("KL by day\n")
        writer.write("{0:10}\n".format("duration"))
        writer.write("KL by day:\n")
        for duration, _ in KLs.iteritems():
            writer.write("{0}\n".format("\t".join(["{0:.4f}".format(kl) for kl in KLs[duration]])))

        writer.write("Average CR:\n")
        writer.write("{0:10}{1:5}\n".format("duration","ACR"))
        for duration, acr in ACRs.iteritems():
            writer.write("{0:2d}\t{1:.4f}\n".format(duration, acr))

        writer.write("CR by day\n")
        writer.write("{0:10}\n".format("duration"))
        for duration, _ in CRs.iteritems():
            writer.write("{0}\n".format("\t".join(["{0:.4f}".format(cr) for cr in CRs[duration]])))
