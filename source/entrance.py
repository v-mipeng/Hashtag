# region Import
import sys

sys.path.append("..")
import datetime
from blocks.extensions import Printing, FinishAfter, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent
import theano
from blocks.graph import ComputationGraph
from dataset import *
from util.entrance import *
from config import *
from util.entrance import *
import logging
import numpy

# endregion

# region Logging
try:
    from blocks_extras.extensions.plot import Plot

    plot_avail = False
except ImportError:
    plot_avail = False
    print "No plotting extension available."

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

sys.setrecursionlimit(2000000)


# endregion


class FD2Entrance(object):
    def __init__(self, config=None):
        self.config = config
        self.dataset = FD2D(self.config)

    def initialize(self, train_raw_dataset, *args, **kwargs):
        '''
        Initialize dataset with given training dataset
        '''
        self.dataset.set_train_data(train_raw_dataset)

    def predict(self, item, *args, **kwargs):
        user = item[self.config.user_index]
        return self.dataset.predict(user)


class BaseEntrance(object):
    '''

    '''

    def __init__(self, config, *args, **kwargs):
        assert config is not None
        self._initialize(config)

    def _initialize(self, config, *args, **kwargs):
        self.config = config
        self.raw_dataset = RUTHD(self.config)
        self.dataset = self.config.Dataset(self.config)
        self.backup_system = FD2Entrance(config)
        self.model = None
        self.model_save_loader = self.config.model_save_loader

    def train(self, *args, **kwargs):
        '''
                Train a user-text-hashtag lstm model with given training dataset or the default dataset which is defined with config.train_path

                @param train_path: path of the training dataset, file or directory, default: config.train_path
                                   File foramt: Mention TAB True_label TAB Context

                @param valid_portion: a float value define the portion of validation, default: config.multi_time_lstm.MLTC.valid_portion
                                      size of validation dataset: all_the_sample_num * valid_portion

                @param valid_path: path of the validation dataset, file or directory, if given, the valid_portion will be 0.


                @param model_path: path to dump the trained model, default: config.multi_time_lstm module.model_path
                '''
        load_from = self.config.model_path
        self.raw_dataset.prepare()
        if self.config.begin_date is None:
            date_offset = 0
        else:
            date_offset = self.config.begin_date
        T = self.config.T
        tmp = self.raw_dataset.get_dataset(reference_date=self.raw_dataset.FIRST_DAY,
                                           date_offset=T - 1,
                                           duration=T)
        self._build_backup_system(tmp)
        numpy.random.seed(123)
        # Splite training and validation set randomly
        rvs = numpy.random.uniform(low=0., high=1., size=len(tmp))
        train_raw_dataset = tmp[rvs > self.config.valid_percent]
        valid_raw_dataset = tmp[rvs <= self.config.valid_percent]
        # train_raw_dataset = self.raw_dataset.get_dataset(reference_date=self.raw_dataset.FIRST_DAY,
        #                                   date_offset=T-2,
        #                                   duration=self.config.time_window)
        # valid_raw_dataset = self.raw_dataset.get_dataset(reference_date=self.raw_dataset.FIRST_DAY,
        #                                                  date_offset=T - 1,
        #                                                  duration=1)
        test_raw_dataset = self.raw_dataset.get_dataset(reference_date=self.raw_dataset.FIRST_DAY,
                                                        date_offset=T,
                                                        duration=1)
        date = self.raw_dataset.first_date + datetime.timedelta(days=T)
        save_to = load_from
        # train_stream = self.dataset.get_train_stream(train_raw_dataset)
        # self.n_samples = len(train_raw_dataset)
        # valid_stream = self.dataset.get_valid_stream(valid_raw_dataset)
        # print("Train on {0} hashtags\n".format(len(self.dataset.hashtag2index.keys())))
        # # Train the model !
        # logging.info("Training model on date:{0} ...".format(date))
        # self._train_model(train_stream, valid_stream, load_from, save_to)
        # logger.info("Training model on date:{0} finished!".format(date))
        # Test model
        test_stream = self.dataset.get_test_stream(test_raw_dataset)
        logging.info("Test model on date:{0} ...".format(date))
        save_to = self.config.test_result_path
        self.test(test_raw_dataset, test_stream, load_from=load_from, save_to=save_to)
        logger.info("Test model on date:{0} finished!".format(date))

    def _build_backup_system(self, train_raw_dataset):
        self.backup_system.initialize(train_raw_dataset)

    def _train_model(self, train_stream, valid_stream, load_from, save_to, *args, **kwargs):
        # Build model
        self.model = self.config.Model(self.config, self.dataset)  # with word2id

        cg = Model(self.model.cg_generator)

        algorithm = GradientDescent(cost=self.model.cg_generator,
                                    step_rule=self.config.step_rule,
                                    parameters=cg.parameters,
                                    on_unused_sources='ignore')

        if plot_avail:
            extensions = [FinishAfter(after_n_epochs=1),
                          TrainingDataMonitoring([v for l in self.model.monitor_train_vars for v in l],
                                                 prefix='train',
                                                 every_n_batches=self.config.print_freq),
                          Plot('Training Process',
                               channels=[v.name for l in self.model.monitor_train_vars for v in l],
                               after_batch=True)]
        else:
            extensions = [
                TrainingDataMonitoring(
                    [v for l in self.model.monitor_train_vars for v in l],
                    prefix='train',
                    every_n_batches=self.config.print_freq)
            ]

        saver_loader = self.model_save_loader(load_from=load_from,
                                              save_to=save_to,
                                              model=cg,
                                              dataset=self.dataset)
        saver_loader.do_load()

        n_batches = numpy.ceil(self.n_samples / self.config.batch_size).astype('int32')
        n_valid_batches = numpy.ceil(n_batches * self.config.valid_freq).astype('int32')
        extensions += [
            EvaluatorWithEarlyStop(
                coverage=1.,
                tolerate_time=self.config.tolerate_time,
                variables=[v for l in self.model.monitor_valid_vars for v in l],
                monitor_variable=self.model.stop_monitor_var,
                data_stream=valid_stream,
                saver=saver_loader,
                prefix='valid',
                every_n_batches=n_valid_batches)
        ]

        extensions += [
            Printing(every_n_batches=self.config.print_freq, after_epoch=True),
            ProgressBar()
        ]
        extensions += [EpochMonitor(1)]
        main_loop = MainLoop(
            model=cg,
            data_stream=train_stream,
            algorithm=algorithm,
            extensions=extensions
        )
        # Run the model!
        main_loop.run()

    def test(self, test_raw_dataset, test_stream, load_from, save_to, *args, **kwargs):
        # Build model
        self.model = self.config.Model(self.config, self.dataset)  # with word2id

        cg = Model(self.model.cg_generator)

        # Initialize model
        initializer = self.model_save_loader(load_from=load_from,
                                             save_to=load_from,
                                             model=cg,
                                             dataset=self.dataset,
                                             before_training=True)
        initializer.do_load()

        # predicted hashtags
        pred_hashtag_idxes = []
        inputs = cg.inputs
        output = self.model.monitor_extra_vars[0]
        f_rank = theano.function(inputs, output, on_unused_input='ignore')
        for batch in test_stream.get_epoch_iterator():
            input_batch = (batch[test_stream.sources.index(input.name)] for input in inputs)
            pred_hashtag_idxes.append(f_rank(*input_batch))
        self._get_result(test_raw_dataset, numpy.concatenate(numpy.array(pred_hashtag_idxes), axis=0), save_to)

    def _get_result(self, test_raw_dataset, pred_hashtag_idxes, save_to):
        keys, values = zip(*self.dataset.hashtag2index.items())
        index2hashtag = dict(zip(values, keys))
        hashtags = []
        for i in range(len(keys)):
            hashtags.append(index2hashtag[i])
        hashtags = numpy.array(hashtags)
        true_hashtags = zip(*test_raw_dataset)[self.config.hashtag_index]
        # Get predict hashtags
        pred_hashtag_idxes = numpy.array(pred_hashtag_idxes)
        shapes = pred_hashtag_idxes.shape
        pred_hashtags = hashtags[pred_hashtag_idxes.reshape(1, -1)].reshape(shapes).tolist()
        backed_pred_hashtags = self._backup_with_FD2(test_raw_dataset, pred_hashtags)
        dic = self._get_statistic_result(true_hashtags, pred_hashtags)
        with open(save_to, 'w+') as writer:
            for true_hashtag, origianl_pred_hashtag, backed_pred_hashtag in zip(true_hashtags, pred_hashtags, backed_pred_hashtags):
                # Save first 1000 predicted hashtags
                writer.write("{0}\t{1}\t{2}\n".format(true_hashtag, origianl_pred_hashtag[0], ";".join(backed_pred_hashtag[0:1000])))
            for item in dic.items():
                writer.write("{0}:{1}\n".format(item[0], item[1]))

    def _backup_with_FD2(self, test_raw_dataset, pred_hashtags):
        assert len(test_raw_dataset) == len(pred_hashtags)
        new_pred_hashtags = []
        for pred_hashtag, item in zip(pred_hashtags, test_raw_dataset):
            if pred_hashtag[0] == '<unk>':
                new_pred_hashtags.append(self.backup_system.predict(item))
            else:
                new_pred_hashtags.append(pred_hashtag)
        return new_pred_hashtags

    def _get_statistic_result(self, true_hashtags, pred_hashtags):
        '''
        Get the predict result on some measurement
        :param true_hashtags:
        :param pred_hashtags:
        :return: A dict object with measure description and corresponding value
        '''
        dic = self._get_result_on_rank1(true_hashtags, pred_hashtags)
        dic.update(self._get_result_on_rank10(true_hashtags, pred_hashtags))
        dic.update(self._get_result_on_mrr(true_hashtags, pred_hashtags))
        return dic

    def _get_result_on_rank1(self, true_hashtags, pred_hashtags):
        assert len(true_hashtags) == len(pred_hashtags)
        count = 0
        for true_hashtag, pred_hashtag in zip(true_hashtags, pred_hashtags):
            if true_hashtag == pred_hashtag[0]:
                count += 1
            else:
                continue
        return {'accuracy_on_rank1': 1. * count / len(true_hashtags)}

    def _get_result_on_rank10(self, true_hashtags, pred_hashtags):
        assert len(true_hashtags) == len(pred_hashtags)
        count = 0
        for true_hashtag, pred_hashtag in zip(true_hashtags, pred_hashtags):
            if true_hashtag in pred_hashtag[0:10]:
                count += 1
            else:
                continue
        return {'accuracy_on_rank10': 1. * count / len(true_hashtags)}

    def _get_result_on_mrr(self, true_hashtags, pred_hashtags):
        '''
        Get result on Mean Reciprocal Rank(MRR)
        :param true_hashtags:
        :param pred_hashtags:
        :return:
        '''
        assert len(true_hashtags) == len(pred_hashtags)
        mrr = 0.
        for true_hashtag, pred_hashtag in zip(true_hashtags, pred_hashtags):
            try:
                index = pred_hashtag.index(true_hashtag)
                mrr += 1. / index
            except:
                continue
        return {'accuracy_on_mse': mrr / len(true_hashtags)}

    def monitor_extra(self, stream, load_from, *args, **kwargs):
        # Monitor Hit@1~10
        self.model = self.config.Model(self.config, self.dataset)  # with word2id

        cg = Model(self.model.cg_generator)

        initializer = self.model_save_loader(load_from=load_from,
                                             save_to=load_from,
                                             model=cg,
                                             dataset=self.dataset,
                                             before_training=True)
        initializer.do_load()

        inputs = cg.inputs
        output = self.model.monitor_extra_vars[0]

        if 'top_n' in kwargs:
            top_n = kwargs['top_n']
        else:
            top_n = 10
        hits = numpy.zeros(top_n, dtype='int32')
        f_rank = theano.function(inputs, output)
        data_size = 0
        for batch in stream.get_epoch_iterator():
            data_size += len(batch[0])
            input_batch = [batch[stream.sources.index(input.name)] for input in inputs]
            ranks = f_rank(input_batch)
            true_hashtag_index = batch[stream.sources.index]
            for i in range(1, top_n + 1):
                hits[i] += (true_hashtag_index[:, None] == ranks[:, 0:i]).sum()
        hits = 1.0 * hits / data_size
        for i in range(1, top_n + 1):
            print("Hit at rank {0}:{1}".format(i, hits[i]))


class FDUTHE(object):
    def __init__(self, config=None):
        self._initialize(config)

    def _initialize(self, config=None, *args, **kwargs):
        if config is None:
            self.config = UTHC
        else:
            self.config = config
        self.raw_dataset = RUTHD(self.config)
        self.dataset = FDUTHD(self.config)

    def do(self, *args, **kwargs):
        T = self.config.T

        # TODO: iterate on date_offset to get dynamic result
        train_raw_dataset = self.raw_dataset.get_dataset(reference_date=self.raw_dataset.FIRST_DAY,
                                                         date_offset=T - 1,
                                                         duration=self.config.time_window)
        test_raw_dataset = self.raw_dataset.get_dataset(reference_date=self.raw_dataset.FIRST_DAY,
                                                        date_offset=T,
                                                        duration=1)
        self.dataset.set_train_data(train_raw_dataset)
        for alpha in range(1, 10):
            self.dataset.set_alpha(0.001 * alpha)
            top1_accuracy, top10_accuracy = self.dataset.test(test_raw_dataset)
            print("top1 accuracy:{0} with alpha={1}".format(top1_accuracy, 0.001 * alpha))
            print("top10 accuracy:{0} with alpha={1}".format(top10_accuracy, 0.001 * alpha))

    def stat(self, *args, **kwargs):
        T = self.config.T

        # TODO: iterate on date_offset to get dynamic result
        train_raw_dataset = self.raw_dataset.get_dataset(reference_date=self.raw_dataset.FIRST_DAY,
                                                         date_offset=T - 1,
                                                         duration=T)
        test_raw_dataset = self.raw_dataset.get_dataset(reference_date=self.raw_dataset.FIRST_DAY,
                                                        date_offset=T,
                                                        duration=1)
        self.dataset.set_train_data(train_raw_dataset)
        fields = zip(*test_raw_dataset)
        test_users = fields[self.config.user_index]
        test_hashtags = fields[self.config.hashtag_index]
        top_n = 10
        alphas = (0, 0.001, 1)
        # alphas = (0,)
        for alpha in alphas:
            self.dataset.set_alpha(alpha)
            print("alpha={0}".format(alpha))
            result = numpy.zeros(10, dtype='int32')
            for user, hashtag in zip(test_users, test_hashtags):
                result += self.dataset.test_top_n(user, hashtag, top_n)
            result_mean = 1.0 * result / len(test_users)
            for value in result_mean:
                print(value)


class TUTHE(object):
    def __init__(self, config=None, *args, **kwargs):
        super(TUTHE, self).__init__(config)

    def test(self, test_stream, load_from, *args, **kwargs):
        load_from = self.config.model_path
        test_raw_dataset = self.raw_dataset.get_dataset(reference_date=self.raw_dataset.FIRST_DAY,
                                                        date_offset=self.config.T,
                                                        duration=1)
        test_stream = self.dataset.get_test_stream(test_raw_dataset)
        # Build model
        self.model = self.config.Model(self.config, self.dataset)  # with word2id

        cg = Model(self.model.cg_generator)

        initializer = self.model_save_loader(load_from=load_from,
                                             save_to=load_from,
                                             model=cg,
                                             dataset=self.dataset,
                                             before_training=True)
        initializer.do_load()

        # self._apply_droput(cg)

        extension = MyDataStreamMonitoring([v for l in self.model.monitor_test_vars for v in l],
                                           test_stream,
                                           coverage=1.,
                                           before_training=True,
                                           prefix='valid')
        date = self.raw_dataset.first_date + datetime.timedelta(days=self.config.T)
        print('Test on date:{0}'.format(date))
        extension.do("before_training")

    #
    # def _apply_droput(self, cg):
    #     params = cg.get_parameter_values()
    #     params['/word_embed.W'] *= 1-self.config.dropout_prob
    #     # params['/user_embed.W'] *= 1-self.config.dropout_prob
    #     params['/hashtag_embed.W'] *= 1-self.config.dropout_prob
    #     cg.set_parameter_values(params)

    def monitor_extra(self, stream, load_from, *args, **kwargs):
        # Monitor Hit@1~10
        if load_from is None:
            load_from = self.config.model_path
        else:
            pass
        if stream is None:
            test_raw_dataset = self.raw_dataset.get_dataset(reference_date=self.raw_dataset.FIRST_DAY,
                                                            date_offset=self.config.T,
                                                            duration=1)
            stream = self.dataset.get_test_stream(test_raw_dataset)
        else:
            pass

        if 'top_n' in kwargs:
            top_n = kwargs['top_n']
        else:
            top_n = 10

        self.model = self.config.Model(self.config, self.dataset)  # with word2id

        cg = Model(self.model.cg_generator)

        initializer = self.model_save_loader(load_from=load_from,
                                             save_to=load_from,
                                             model=cg,
                                             dataset=self.dataset,
                                             before_training=True)
        initializer.do_load()

        inputs = cg.inputs
        output = self.model.monitor_extra_vars[0]

        hits = numpy.zeros(top_n, dtype='int32')
        f_rank = theano.function(inputs, output, on_unused_input='ignore')
        data_size = 0
        for batch in stream.get_epoch_iterator():
            data_size += len(batch[0])
            input_batch = (batch[stream.sources.index(input.name)] for input in inputs)
            ranks = f_rank(*input_batch)
            true_hashtag_index = batch[stream.sources.index('hashtag')]
            for i in range(1, top_n + 1):
                hits[i - 1] += (true_hashtag_index[:, None] == ranks[:, 0:i]).sum()
        hits = 1.0 * hits / data_size
        for i in range(1, top_n + 1):
            print("Hit at rank {0}:{1}".format(i, hits[i - 1] * self.dataset.hashtag_coverage))


if __name__ == "__main__":
    # config = AttentionEUTHC
    # config.model_path = os.path.join(config.project_dir,"output/model/AEUTH/AEUTH_2015-01-11.pkl")
    # entrance = TUTHE(config)
    # entrance.monitor_extra(None,None)

    # config = AttentionEUTHC2
    # config.model_path = os.path.join(config.project_dir,"output/model/AEUTH2/AEUTH2_2015-01-11.pkl")
    # entrance = TUTHE(config)
    # entrance.monitor_extra(None, None)


    # config = UHC
    # config.model_path = os.path.join(config.project_dir, "output/model/UH/UH_2015-01-11.pkl")
    # entrance = TUTHE(config)
    # entrance.monitor_extra(None, None)

    # config = AttentionEUTHC
    # config.model_path = os.path.join(config.project_dir, "output/model/RAEUTH/RAEUTH_2015-01-11.pkl")
    # config.Model = UHM2
    # entrance = TUTHE(config)
    # entrance.monitor_extra(None, None)


    #
    # config = ETHC
    # config.model_path = os.path.join(config.project_dir,"output/model/RETH/RETH_2015-01-11.pkl")
    # entrance = TUTHE(config)
    # entrance.monitor_extra(None,None)
    #

    config = AttentionEUTHC
    config.model_path = os.path.join(config.project_dir, "output/sig vs com/RCAEUTH/RCAEUTH.pkl")
    entrance = TUTHE(config)
    entrance.monitor_extra(None, None)
