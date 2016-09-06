#region Import
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
#endregion

#region Logging
try:
    from blocks_extras.extensions.plot import Plot
    plot_avail = False
except ImportError:
    plot_avail = False
    print "No plotting extension available."

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

sys.setrecursionlimit(2000000)
#endregion


class UTHE(object):
    '''

    '''

    def __init__(self, config = None):
        self._initialize(config)

    def _initialize(self, config = None):
        if config is None:
            self.config = UTHC
        else:
            self.config = config
        self.raw_dataset = RUTHD(self.config)
        self.dataset = UTHD(self.config)
        self.model = None
        self.model_save_loader = BasicSaveLoadParams

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
        model_base_name, _ = os.path.splitext(self.config.model_path)
        count = 0
        self.raw_dataset.prepare()
        if self.config.begin_date is None:
            date_offset = 0
        else:
            date_offset = self.config.begin_date
        # TODO: iterate on date_offset to get dynamic result
        tmp = self.raw_dataset.get_dataset(reference_date=self.raw_dataset.FIRST_DAY,
                                           date_offset=date_offset + self.config.time_window - 1,
                                           duration=self.config.time_window)
        numpy.random.seed(123)
        rvs = numpy.random.uniform(low=0., high=1., size=len(tmp))
        train_raw_dataset = tmp[rvs > self.config.valid_percent]
        valid_raw_dataset = tmp[rvs <= self.config.valid_percent]
        test_raw_dataset = self.raw_dataset.get_dataset(reference_date=self.raw_dataset.FIRST_DAY,
                                                        date_offset=date_offset + self.config.time_window,
                                                        duration=1)
        date = self.raw_dataset.first_date + datetime.timedelta(days=date_offset + self.config.time_window)
        save_to = "{0}_{1}.pkl".format(model_base_name, str(date))
        train_stream = self.dataset.get_train_stream(train_raw_dataset)
        valid_stream = self.dataset.get_test_stream(valid_raw_dataset)
        print("Train on {0} hashtags\n".format(len(self.dataset.hashtag2index.keys())))
        # Train the model !
        logging.info("Training model on date:{0} ...".format(date))
        self._train_model(train_stream, valid_stream, load_from, save_to)
        logger.info("Training model on date:{0} finished!".format(date))
        test_stream = self.dataset.get_test_stream(test_raw_dataset)
        logging.info("Test model on date:{0} ...".format(date))
        self.test(test_stream, load_from=load_from)
        logger.info("Test model on date:{0} finished!".format(date))

    def _train_model(self, train_stream, valid_stream, load_from, save_to):
        # Build model
        self.model = self.config.Model(self.config, self.dataset)  # with word2id

        cg = Model(self.model.cg_generator)

        params = cg.get_parameter_values()

        algorithm = GradientDescent(cost=self.model.cg_generator,
                                    step_rule=self.config.step_rule,
                                    parameters=cg.parameters,
                                    on_unused_sources='ignore')

        if plot_avail:
            extensions = [FinishAfter(after_n_epochs=1),
                          TrainingDataMonitoring([v for l in self.model.monitor_vars for v in l],
                                                 prefix='train',
                                                 every_n_batches=self.config.print_freq),
                          Plot('Training Process',
                               channels=[[self.model.monitor_train_vars[0][0].name],
                                         [self.model.monitor_train_vars[1][0].name]],
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

        extensions += [
            EvaluatorWithEarlyStop(
                coverage=self.dataset.hashtag_coverage,
                tolerate_time=self.config.tolerate_time,
                variables=[v for l in self.model.monitor_valid_vars for v in l],
                monitor_variable=self.model.stop_monitor_var,
                data_stream=valid_stream,
                saver=saver_loader,
                prefix='valid',
                every_n_epochs=self.config.valid_freq)
        ]

        extensions += [
            Printing(every_n_batches=self.config.print_freq, after_epoch=True),
            ProgressBar()
        ]
        main_loop = MainLoop(
            model=cg,
            data_stream=train_stream,
            algorithm=algorithm,
            extensions=extensions
        )
        # Run the model !
        main_loop.run()

    def test(self, test_stream, load_from):
        # Build model
        self.model = self.config.Model(self.config, self.dataset)  # with word2id

        cg = Model(self.model.cg_generator)
        saver_loader = self.model_save_loader(load_from=load_from,
                                              save_to=None,
                                              model=cg,
                                              dataset=self.dataset)
        saver_loader.do_load()
        inputs = cg.inputs
        top1_recall = 0.
        top10_recall = 0.
        f_top1 = theano.function(inputs, self.model.monitor_valid_vars[0][0])
        f_top10 = theano.function(inputs, self.model.monitor_valid_vars[1][0])
        data_size = 0
        for batch in test_stream.get_epoch_iterator():
            data_size += batch[0].shape[0]
            stream_inputs = []
            for input in inputs:
                stream_inputs.append(batch[test_stream.sources.index(input.name)])
            top1_recall += f_top1(*tuple(stream_inputs)) * stream_inputs[0].shape[0]
            top10_recall += f_top10(*tuple(stream_inputs)) * stream_inputs[0].shape[0]
        top1_recall /= data_size
        top10_recall /= data_size
        print("Test hashtag coverage:{0}".format(self.dataset.hashtag_coverage))
        print("top1_recall:{0}\n".format(top1_recall * self.dataset.hashtag_coverage))
        print("top10_recall:{0}\n".format(top10_recall * self.dataset.hashtag_coverage))


class BiasUTHE(UTHE):
    def __init__(self, config = None):
        super(BiasUTHE, self).__init__(config)

    def _initialize(self, config = None):
        if config is None:
            self.config = BiasUTHC
        else:
            self.config = config
        super(BiasUTHE, self)._initialize(config)
        self.model_save_loader = BiasSaveLoadParams


class EUTHE(UTHE):
    '''Train Extended UTH model which apply hashtag bias'''
    def __init__(self, config = None):
        super(EUTHE, self).__init__(config)

    def _initialize(self, config = None):
        if config is None:
            self.config = EUTHC
        else:
            self.config = config
        self.raw_dataset = RUTHD(self.config)
        self.dataset = EUTHD(self.config)
        self.model = None
        self.model_save_loader = ExtendSaveLoadParams


class OVHashtagUTHE(BiasUTHE):
    def __init__(self, config = None):
        super(OVHashtagUTHE, self).__init__(config)

    def _initialize(self, config = None):
        if config is None:
            self.config = UTHC
        else:
            self.config = config
        self.raw_dataset = RUTHD(self.config)
        self.dataset = OVHashtagUTHD(self.config)
        self.model = None
        self.model_save_loader = BiasSaveLoadParams


class EMicroblogTHE(EUTHE):
    def __init__(self, config = None):
        super(EMicroblogTHE, self).__init__(config)

    def _initialize(self, config = None):
        if config is None:
            self.config = EMicroblogTHC
        else:
            self.config = config
        self.raw_dataset = RMicroblogDataset(self.config)
        self.dataset = EMicroblogTHD(self.config)
        self.model = None
        self.model_save_loader = ExtendSaveLoadParams


    def train(self, *args, **kwargs):
        load_from = self.config.model_path
        model_base_name, _ = os.path.splitext(self.config.model_path)
        count = 0
        self.raw_dataset.prepare()
        # TODO: iterate on date_offset to get dynamic result
        tmp = numpy.array(self.raw_dataset.get_dataset())
        numpy.random.seed(123)
        rvs = numpy.random.uniform(low=0., high=1., size=len(tmp))
        valid_raw_dataset = tmp[rvs <= self.config.valid_percent]
        test_raw_dataset = tmp[numpy.logical_and(rvs > self.config.valid_percent, rvs <= self.config.valid_freq+self.config.test_percent)]
        train_raw_dataset = tmp[rvs > self.config.valid_percent+self.config.test_percent]
        save_to = load_from
        train_stream = self.dataset.get_train_stream(train_raw_dataset)
        valid_stream = self.dataset.get_test_stream(valid_raw_dataset)
        print("Train on {0} hashtags\n".format(len(self.dataset.hashtag2index.keys())))
        # Train the model !
        logging.info("Training model ...")
        self._train_model(train_stream, valid_stream, load_from, save_to)
        logger.info("Training model finished!")
        test_stream = self.dataset.get_test_stream(test_raw_dataset)
        logging.info("Test model on date:{0} ...")
        self.test(test_stream, load_from=load_from)
        logger.info("Test model on date:{0} finished!")


class TimeLineUTHE(UTHE):
    #TODO: complete this class
    def __init__(self, config = None):
        super(TimeLineUTHE, self).__init__(config)

    def _initialize(self, config = None):
        if config is None:
            self.config = UTHC
        else:
            self.config = config
        self.raw_dataset = RUTHD(self.config)
        self.dataset = TimeLineUTHD(self.config)
        self.model = None
        self.model_save_loader = BasicSaveLoadParams

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
        model_base_name, _ = os.path.splitext(self.config.model_path)
        count = 0
        self.raw_dataset.prepare()
        if self.config.begin_date is None:
            date_offset = 0
        elif isinstance(self.config.begin_date, datetime.date):
            if self.config.begin_date >= self.raw_dataset.first_date:
                date_offset = (self.config.begin_date - self.raw_dataset.first_date).days
            else:
                raise ValueError('begin date should be latter than the earlist date of the dataset')
        else:
            date_offset = self.config.begin_date
        #TODO: iterate on date_offset to get dynamic result
        for i in range(self.config.time_window):
            date = self.raw_dataset.first_date + datetime.timedelta(days = date_offset+i+1)
            save_to = "{0}_{1}.pkl".format(model_base_name, str(date))
            if i < self.config.time_window-1:
                train_raw_dataset = self.raw_dataset.get_dataset(reference_date= self.raw_dataset.FIRST_DAY,
                                                            date_offset = date_offset+i,
                                                            duration = 1)
                valid_raw_dataset = self.raw_dataset.get_dataset(reference_date= self.raw_dataset.FIRST_DAY,
                                                            date_offset = date_offset+i+1,
                                                            duration = 1)
            else:
                tmp = self.raw_dataset.get_dataset(reference_date= self.raw_dataset.FIRST_DAY,
                                                            date_offset = date_offset+i,
                                                            duration = 1)
                rvs = numpy.random.uniform(low=0., high=1., size=len(tmp))
                train_raw_dataset = tmp[rvs > self.config.valid_percent]
                valid_raw_dataset = tmp[rvs <= self.config.valid_percent]
            train_stream = self.dataset.get_train_stream(train_raw_dataset)
            valid_stream = self.dataset.get_test_stream(valid_raw_dataset)
            print("Train on {0} hashtags\n".format(len(self.dataset.hashtag2index.keys())))
            # Train the model !
            logging.info("Training model on date:{0} ...".format(date))
            self._train_model(train_stream, valid_stream, load_from, save_to)
            logger.info("Training model on date:{0} finished!".format(date))
            load_from = save_to
            if i == self.config.time_window-1:
                test_raw_dataset = self.raw_dataset.get_dataset(reference_date= self.raw_dataset.FIRST_DAY,
                                                            date_offset = date_offset+i+1,
                                                            duration = 1)
                test_stream = self.dataset.get_test_stream(test_raw_dataset)
                logging.info("Test model on date:{0} ...".format(date))
                self.test(test_stream, load_from = load_from)
                logger.info("Test model on date:{0} finished!".format(date))

    def _train_model(self, train_stream, valid_stream, load_from, save_to):
        # Build model
        self.model = self.config.Model(self.config, self.dataset)  # with word2id

        cg = Model(self.model.cg_generator)

        params = cg.get_parameter_values()

        algorithm = GradientDescent(cost=self.model.cg_generator,
                                    step_rule=self.config.step_rule,
                                    parameters=cg.parameters,
                                    on_unused_sources='ignore')

        if plot_avail:
            extensions = [FinishAfter(after_n_epochs=1),
                          TrainingDataMonitoring([v for l in self.model.monitor_vars for v in l],
                                                 prefix='train',
                                                 every_n_batches=self.config.print_freq),
                          Plot('Training Process',
                               channels=[[self.model.monitor_train_vars[0][0].name],
                                         [self.model.monitor_train_vars[1][0].name]],
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

        extensions += [
            EvaluatorWithEarlyStop(
                coverage=self.dataset.hashtag_coverage,
                tolerate_time=self.config.tolerate_time,
                variables=[v for l in self.model.monitor_valid_vars for v in l],
                monitor_variable=self.model.stop_monitor_var,
                data_stream=valid_stream,
                saver=saver_loader,
                prefix='valid',
                every_n_epochs=self.config.valid_freq)
        ]

        extensions += [
            Printing(every_n_batches=self.config.print_freq, after_epoch=True),
            ProgressBar()
        ]
        main_loop = MainLoop(
            model=cg,
            data_stream=train_stream,
            algorithm=algorithm,
            extensions=extensions
        )
        # Run the model !
        main_loop.run()

    def test(self, test_stream, load_from):
        # Build model
        self.model = self.config.Model(self.config, self.dataset)  # with word2id

        cg = Model(self.model.cg_generator)
        saver_loader = self.model_save_loader(load_from=load_from,
                                              save_to=None,
                                              model=cg,
                                              dataset=self.dataset)
        saver_loader.do_load()
        inputs = cg.inputs
        top1_recall = 0.
        top10_recall = 0.
        f_top1 = theano.function(inputs, self.model.monitor_valid_vars[0][0])
        f_top10 = theano.function(inputs, self.model.monitor_valid_vars[1][0])
        data_size = 0
        for batch in test_stream.get_epoch_iterator():
            data_size += batch[0].shape[0]
            stream_inputs = []
            for input in inputs:
                stream_inputs.append(batch[test_stream.sources.index(input.name)])
            top1_recall += f_top1(*tuple(stream_inputs)) * stream_inputs[0].shape[0]
            top10_recall += f_top10(*tuple(stream_inputs)) * stream_inputs[0].shape[0]
        top1_recall /= data_size
        top10_recall /= data_size
        print("Test hashtag coverage:{0}".format(self.dataset.hashtag_coverage))
        print("top1_recall:{0}\n".format(top1_recall*self.dataset.hashtag_coverage))
        print("top10_recall:{0}\n".format(top10_recall*self.dataset.hashtag_coverage))


class TimeLineEUTHE(TimeLineUTHE):
    def __init__(self, config = None):
        super(TimeLineEUTHE, self).__init__(config)

    def _initialize(self, config = None):
        if config is None:
            self.config = FUTHC
        else:
            self.config = config
        self.raw_dataset = RUTHD(self.config)
        self.dataset = TimeLineEUTHD(self.config)
        self.model = None
        self.model_save_loader = ExtendSaveLoadParams


class TUTHE(EUTHE):

    def __init__(self, config = None):
        super(TUTHE,self).__init__(config)

    def test(self):
        load_from = self.config.model_path
        test_stream = self.dataset.get_shuffled_stream_by_date(self.config.begin_date, update=False)
        # Build model
        self.model = self.config.Model(self.config, self.dataset)  # with word2id

        cg = Model(self.model.cg_generator)
        initializer = BasicSaveLoadParams(load_from=load_from,
                                   save_to = load_from,
                                   model=cg,
                                   dataset= self.dataset,
                                   before_training=True)
        initializer.do_load()
        inputs = cg.inputs
        top1_recall = 0.
        top10_recall = 0.
        f_top1 = theano.function(inputs, self.model.top1_recall)
        f_top10 = theano.function(inputs, self.model.top10_recall)
        data_size = 0
        for batch in test_stream.get_epoch_iterator():
            data_size += batch[0].shape[0]
            stream_inputs = []
            for input in inputs:
                stream_inputs.append(batch[test_stream.sources.index(input.name)])
            top1_recall += f_top1(*tuple(stream_inputs))*stream_inputs[0].shape[0]
            top10_recall += f_top10(*tuple(stream_inputs))*stream_inputs[0].shape[0]
        top1_recall /= data_size
        top10_recall /= data_size
        print("top1_recall:{0}\n".format(top1_recall))
        print("top10_recall:{0}\n".format(top10_recall))


if __name__ ==  "__main__":
    config = TUTHC
    config.model_path = os.path.join(config.project_dir, "output/model/UTHC_lstm_2015-01-31.pkl")
    config.begin_date = datetime.date(2015, 2, 11)
    entrance = TUTHE(config)
    entrance.test()
    rvg = numpy.random.seed(123)