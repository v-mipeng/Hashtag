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
from dataset import SUTHD, NUTHD, TUTHD, FUTHD, RUTHD
from util.entrance import *
from config import UTHC, EUTHC, TUTHC, FUTHC
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
        self.dataset = NUTHD(self.config)
        self.iter_dataset = SUTHD(self.config, self.dataset)
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
        model_base_name, _= os.path.splitext(self.config.model_path)
        count = 0
        for train_stream, valid_stream, date in self.iter_dataset:
            print("Train on {0} hashtags on date:{1}\n".format(len(self.dataset.hashtag2index.keys()), date))
            # Build model
            save_to = "{0}_{1}.pkl".format(model_base_name, str(date))
            self.model = self.config.Model(self.config, self.dataset)  # with word2id

            cg = Model(self.model.cg_generator)

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
                                   save_to = save_to,
                                   model=cg,
                                   dataset= self.dataset)
            saver_loader.do_load()

            extensions += [
                EvaluatorWithEarlyStop(
                    coverage=1.,
                    variables = [v for l in self.model.monitor_valid_vars for v in l],
                    monitor_variable=self.model.monitor_valid_vars[0][0],
                    data_stream=valid_stream,
                    saver=saver_loader,
                    prefix='valid',
                    every_n_epochs=self.config.valid_freq)
            ]

            # if self.config.save_freq is not None:
            #     extensions += [
            #         self.model_save_loader(load_from=load_from,
            #                        save_to = save_to,
            #                        model=cg,
            #                        dataset= self.dataset,
            #                        before_training=True,
            #                        after_training=True,
            #                        after_epoch=True,
            #                        every_n_batches=self.config.save_freq)
            #     ]

            extensions += [
                Printing(every_n_batches=self.config.print_freq, after_epoch=True),
                ProgressBar()
            ]

            # if count < 10:
            #     extensions += [EpochMonitor(10)]
            # else:
            #     extensions += [EpochMonitor(30)]
            count += 1
            main_loop = MainLoop(
                model=cg,
                data_stream=train_stream,
                algorithm=algorithm,
                extensions=extensions
            )
            # Run the model !
            logging.info("Training model on date:{0} ...".format(date))
            main_loop.run()
            load_from = "{0}_{1}.pkl".format(model_base_name, str(date))
            logger.info("Training model on date:{0} finished!".format(date))

    def test(self, *args, **kwargs):
        raise NotImplementedError('subclasses must override test()!')

    def predict(self, *args, **kwargs):
        raise NotImplementedError('subclasses must override predict()!')


class SFUTHE(UTHE):
    def __init__(self, config = None):
        super(SFUTHE, self).__init__(config)

    def _initialize(self, config = None):
        if config is None:
            self.config = FUTHC
        else:
            self.config = config
        self.dataset = FUTHD(self.config)
        self.iter_dataset = SUTHD(self.config, self.dataset)
        self.model = None
        self.model_save_loader = ExtendSaveLoadParams

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
        for train_stream, valid_stream, date in self.iter_dataset:
            print("Train on {0} hashtags on date:{1}\n".format(len(self.dataset.hashtag2index.keys()), date))
            # Build model
            save_to = "{0}_{1}.pkl".format(model_base_name, str(date))
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
                    tolerate_time=10,
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

            # if count < 10:
            #     extensions += [EpochMonitor(10)]
            # else:
            #     extensions += [EpochMonitor(30)]
            count += 1
            main_loop = MainLoop(
                model=cg,
                data_stream=train_stream,
                algorithm=algorithm,
                extensions=extensions
            )
            # Run the model !
            logging.info("Training model on date:{0} ...".format(date))
            main_loop.run()
            load_from = "{0}_{1}.pkl".format(model_base_name, str(date))
            logger.info("Training model on date:{0} finished!".format(date))


class FUTHE(UTHE):
    def __init__(self, config = None):
        super(FUTHE, self).__init__(config)

    def _initialize(self, config = None):
        if config is None:
            self.config = FUTHC
        else:
            self.config = config
        self.model = None
        self.model_save_loader = ExtendSaveLoadParams

    def _split_train_valid(self, sample_percent = 0.2):
        raw_dataset = RUTHD(self.config)
        raw_dataset.prepare()
        dataset_in_window = raw_dataset.get_dataset(reference_date= self.dataset.raw_dataset.FIRST_DAY,
                                                        date_offset = self.config.time_window-1,
                                                        duration = self.config.time_window)
        rvs = numpy.random.uniform(low=0., high=1., size = len(raw_dataset.raw_dataset))
        train_sets = dataset_in_window[rvs > sample_percent]
        valid_sets = dataset_in_window[rvs <= sample_percent]

        self.train_raw_dataset = RUTHD(self.config)
        self._init_dataset(self.train_raw_dataset, train_sets)
        self.valid_raw_dataset = RUTHD(self.config)
        self._init_dataset(self.valid_raw_dataset, valid_sets)

    def _get_hashtag_coverage(self, train_sets, valid_sets):
        train_hashtags =

    def _init_dataset(self, dataset_obj, raw_dataset):
        # Initialize self.date_span, self.first_day, self.last_day
        dataset_obj.raw_dataset = raw_dataset
        dates = numpy.array(zip(*raw_dataset)[self.config.date_index])
        dataset_obj.first_date = dates.min()
        dataset_obj.last_date = dates.max()
        dataset_obj.date_span = (dataset_obj.last_day-dataset_obj.first_day).days + 1

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
        self._split_train_valid()
        self.train_dataset = FUTHD(self.config, self.train_raw_dataset)
        self.valid_dataset = FUTHD(self.config, self.valid_raw_dataset)
        load_from = self.config.model_path
        model_base_name, _ = os.path.splitext(self.config.model_path)
        count = 0
        train_stream = self.train_dataset.get_shuffled_stream(reference_date= self.dataset.raw_dataset.FIRST_DAY,
                                                        date_offset = self.config.time_window-1,
                                                        duration = self.config.time_window,
                                                        update= True)
        test_stream = self.dataset.get_shuffled_stream(reference_date= self.dataset.raw_dataset.FIRST_DAY,
                                                        date_offset = self.config.time_window,
                                                        duration = 1,
                                                        update = False)
        print("Train on {0} hashtags on date:{1}\n".format(len(self.dataset.hashtag2index.keys()), date))
        # Build model
        save_to = self.config.model_path
        self.model = self.config.Model(self.config, self.dataset)

        cg = Model(self.model.cg_generator)

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
                tolerate_time=10,
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

        # if count < 10:
        #     extensions += [EpochMonitor(10)]
        # else:
        #     extensions += [EpochMonitor(30)]
        count += 1
        main_loop = MainLoop(
            model=cg,
            data_stream=train_stream,
            algorithm=algorithm,
            extensions=extensions
        )
        # Run the model !
        logging.info("Training model on date:{0} ...".format(date))
        main_loop.run()
        load_from = "{0}_{1}.pkl".format(model_base_name, str(date))
        logger.info("Training model on date:{0} finished!".format(date))

class TUTHE(UTHE):

    def __init__(self, config = None):
        super(TUTHE,self).__init__(config)

    def _initialize(self, config = None):
        if config is None:
            self.config = TUTHC()
        else:
            self.config = config
        self.dataset = TUTHD(self.config)
        self.model = None

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