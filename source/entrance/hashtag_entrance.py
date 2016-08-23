import logging
import numpy
import sys
sys.path.append("..")
import os

import theano
from theano import tensor

from blocks.extensions import Printing, SimpleExtension, FinishAfter, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent

from dataset.hashtag_dataset import  BUTHD
from paramsaveload import SaveLoadParams

from config.hashtag_config import UTHC

from abc import abstractmethod, ABCMeta

from base import *


try:
    from blocks_extras.extensions.plot import Plot
    plot_avail = False
except ImportError:
    plot_avail = False
    print "No plotting extension available."

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

sys.setrecursionlimit(2000000)


#region Reference Entrance
class BasicEntrance(object):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def train(self, train_path = None, valid_portion = None, model_path = None):
        raise NotImplementedError('subclasses must override train()!')

    @abstractmethod
    def test(self, test_path = None, test_result_path = None, model_path = None):
        raise NotImplementedError('subclasses must override test()!')

    @abstractmethod
    def predict(self, predict_path = None, predict_result_path = None, model_path = None):
        raise NotImplementedError('subclasses must override predict()!')

class MTLE(BasicEntrance):
    '''
    Multiple Time LSTM Entrance
    '''
    def __init__(self):
        self.config = None
        self.dataset = None
        self.model = None
        self.model_path = None
        self.f_pred = None
        self.m = None
        self.id2label = {
                    0:"other",
                    1:"location",
                    2:"organization",
                    3:"person",
                    4:"product"
                }
        self.f_pred_prob = None
        self.pred_inputs = None
        self.init()
    
    def init(self):
        self.config = MTLC()
        self.dataset = MTL(self.config)
        if self.dataset.word2id is None:
            self.init_model()

    def train(self, train_path = None, valid_portion = None, valid_path =None, model_path = None):
        '''
        Train a multi_time_lstm model with given training dataset or the default dataset which is defined with config.multi_time_lstm.BasicConfig.train_path

        @param train_path: path of the training dataset, file or directory, default: config.multi_time_lstm.BasicConfig.train_path
                           File foramt: Mention TAB True_label TAB Context

        @param valid_portion: a float value define the portion of validation, default: config.multi_time_lstm.MLTC.valid_portion
                              size of validation dataset: all_the_sample_num * valid_portion

        @param valid_path: path of the validation dataset, file or directory, if given, the valid_portion will be 0.
                           

        @param model_path: path to dump the trained model, default: config.multi_time_lstm module.model_path
        '''
        if train_path is None:
            train_path = self.config.train_path
        if valid_portion is None:
            valid_portion = self.config.valid_portion
        if model_path is None:
            model_path = self.config.model_path
            self.model_path = model_path
        assert valid_portion >= 0 and valid_portion < 1.0

        if valid_path is None:
            train_stream, valid_stream = self.dataset.get_train_stre
            am(train_path, valid_portion)
        else:
            train_stream = self.dataset.get_train_stream(train_path, 0.0)
            valid_stream = self.dataset.get_train_stream(valid_path, 0.0)
            
        # Build the Blocks stuff for training
        if self.m is None:
            self.m = self.config.Model(self.config, self.dataset) # with word2id
        if self.model is None:
            self.model = Model(self.m.sgd_cost) 

        algorithm = GradientDescent(cost=self.m.sgd_cost,
                                    step_rule=self.config.step_rule,
                                    parameters=self.model.parameters,
                                    on_unused_sources='ignore')
        extensions = [
        TrainingDataMonitoring(
            [v for l in self.m.monitor_vars for v in l],
            prefix='train',
            every_n_batches= self.config.print_freq)
            ]

        if self.config.save_freq is not None and model_path is not None:
            extensions += [
                SaveLoadParams(path=model_path,
                                model=self.model,
                                before_training=True,    # if exist model, the program will load it first
                                after_training=True,
                                after_epoch=True,
                                every_n_batches=self.config.save_freq)
            ]
        if valid_stream is not None and self.config.valid_freq != -1:
            extensions += [
                DataStreamMonitoring(
                    [v for l in self.m.monitor_vars_valid for v in l],
                    valid_stream,
    #                before_first_epoch = False,
                    prefix='valid',
                    every_n_batches=self.config.valid_freq),
            ]
        extensions += [
                Printing(every_n_batches=self.config.print_freq, after_epoch=True),
                ProgressBar()
        ]

        main_loop = MainLoop(
            model=self.model,
            data_stream=train_stream,
            algorithm=algorithm,    # learning algorithm: AdaDelta, Momentum or others
            extensions=extensions
        )
        # Run the model !
        main_loop.run()

    def test(self, test_path = None, test_result_path = None, model_path = None):
        '''
        Test with trained multi_time_lstm model on given test dataset

        @param test_path: path of test dataset, file or directory, default: config.multi_time_lstm.BasicConfig.test_path
                          File foramt: Mention TAB True_label TAB Context

        @param test_result_path: path of file or directory to store the test resultk,if not given, the config.multi_time_lstm module.test_result_path will be used
                                 File format: Mention TAB True_label TAB Predict_label TAB Context

        @param model_path: path to load the trained model, default: config.multi_time_lstm.BasicConfig.model_path
        '''
        # Initilize model
        if model_path is not None:
            if self.model_path is None or model_path != self.model_path:
                self.init_model(model_path)
        elif self.model_path is None:
            self.init_model(self.config.model_path)

        # Test
        if test_path is None:
            test_path = self.config.test_path
        if test_result_path is None:
            test_result_path = self.config.test_result_path
        test_files, test_result_files = get_in_out_files(test_path, test_result_path)
        for test_file, test_result_file in zip(test_files,test_result_files):
            print("Test on %s..." % test_file)
            results = self.pred_by_file(test_file, for_test = True)
            save_result(test_result_file, results)
            print("Done!")

    def predict(self, predict_path = None, predict_result_path = None, model_path = None):
        '''
        Predicte with trained multi_time_lstm model on given predict dataset
        Output file format: Mention TAB Predict_label TAB Context

        @param predict_path: path of predict dataset, file or directory, default: config.multi_time_lstm.BasicConfig.predict_path
                             File foramt: Mention TAB Context

        @param predict_result_path: path of file or directory to store the predict result, default: config.multi_time_lstm.BasicConfig.predict_result_path
                                    File foramt: Mention TAB Predict_label TAB Context

        @param model_path: path to load the trained model, default: config.multi_time_lstm.BasicConfig.model_path 
        '''
        # Initilize model
        if model_path is not None:
            if self.model_path is None or model_path != self.model_path:
                self.init_model(model_path)
        elif self.model_path is None:
            self.init_model(self.config.model_path)
        # Predict
        if predict_path is None:
            predict_path = self.config.predict_path
        if predict_result_path is None:
            predict_result_path = self.config.predict_result_path
        predict_files, predict_result_files = get_in_out_files(predict_path, predict_result_path)
        for predict_file, predict_result_file in zip(predict_files,predict_result_files):
            print("Predict on %s..." % predict_file)
            results = self.pred_by_file(predict_file, for_test = False)
            save_result(predict_result_file, results)
            print("Done!")

    def predict_by_tuples(self, samples, model_path = None):
        '''predict type of given samples.
        This is an API for service

        @param samples: a list of tuples with domains: mention [mention_begin] context. if given, data_path will be ignored
        '''
        # Initilize model
        if model_path is not None:
            if self.model_path is None or model_path != self.model_path:
                self.init_model(model_path)
        elif self.model_path is None:
            self.init_model(self.config.model_path)
        stream, data, errors = self.dataset.get_predict_stream(samples = samples)
        if stream is None:
            return [], errors
        labels, confidences = self.pred(stream)
        if len(labels) == len(errors):
            return labels, confidences
        else:
            labels_copy = []
            confidences_copy = []
            i = 0
            j = 0
            for j in range(len(errors)):
                if errors[j] is True:
                    labels_copy.append('UNKNOWN')
                    confidences_copy.append(0.0)
                else:
                    labels_copy.append(labels[i])
                    confidences_copy.append(confidences[i])
                    i += 1
        return labels_copy, confidences_copy      

    def pred_by_file(self, file_path, for_test = True):
        '''
        Make prediction on the samples within given input_file

        @param output_file: result file path, if it is given, write out the predict result into this file

        @param for_test: boolean value, if true, the output format: mention true_label predict_label context
                            otherwise: mention predict_label context

        @return result: a list of tuples, with every tuple consistent with the output format.
        '''
        if for_test:
            stream, data, errors = self.dataset.get_test_stream(file_path)
        else:
            stream, data, errors = self.dataset.get_predict_stream(file_path)
        result = []
        labels,_ = self.pred(stream)
        data = zip(*data)
        if for_test:
            true_labels = [self.id2label[label_id] for label_id in data[-3]]
            result = zip(data[-2],true_labels, labels, data[-1])
        else:
            result = zip(data[-2], labels, data[-1])
        return result    

    def pred(self, stream):
        '''
        Make prediction on given samples 

        @param stream: class BasicDataset. 
        input datastream

        @return result: a list of types
        '''
        labels = []
        confidences = []
        for inputs in stream.get_epoch_iterator():
            p_inputs = tuple([inputs[stream.sources.index(str(input_name))] for input_name in self.pred_inputs]) 
            label_ids = self.f_pred(*p_inputs)
            confidences += [confidence for confidence in self.f_pred_prob(*p_inputs)]
            labels += [self.id2label[label_id] for label_id in label_ids]
        return labels, confidences   

    def init_model(self, model_path = None):
        if self.m is None:
            self.m = self.config.Model(self.config, self.dataset)
        model = Model(self.m.sgd_cost)   
        if model_path is None:
            model_path = self.config.model_path
        initializer = SaveLoadParams(model_path, model)
        initializer.do_load()
        self.model = model
        self.model_path = model_path
        if self.f_pred is None:
            cg = ComputationGraph(self.m.pred)
            self.pred_inputs = cg.inputs
            self.f_pred = theano.function(self.pred_inputs, self.m.pred)  
            self.f_pred_prob = theano.function(self.pred_inputs, self.m.pred_prob)

class MTLDE(MTLE):
    '''
    Multiple Time LSTM  with DBpedia Entrance
    '''
    def __init__(self):
        super(MTLDE, self).__init__()

    def init(self):
        self.config = MTLDC()
        self.dataset = MTLD(self.config)

class WLSTME(MTLE):
    '''
    Weighted Single LSTM Entrance
    '''
    def __init__(self):
        return super(WLSTME, self).__init__()

    def train(self, train_path = None, valid_portion = None, valid_path =None, model_path = None):
        '''
        Train a multi_time_lstm model with given training dataset or the default dataset which is defined with config.multi_time_lstm.BasicConfig.train_path

        @param train_path: path of the training dataset, file or directory, default: config.multi_time_lstm.BasicConfig.train_path
                           File foramt: Mention TAB True_label TAB Context

        @param valid_portion: a float value define the portion of validation, default: config.multi_time_lstm.MLTC.valid_portion
                              size of validation dataset: all_the_sample_num * valid_portion

        @param valid_path: path of the validation dataset, file or directory, if given, the valid_portion will be 0.
                           

        @param model_path: path to dump the trained model, default: config.multi_time_lstm module.model_path
        '''
        if train_path is None:
            train_path = self.config.train_path
        if valid_portion is None:
            valid_portion = self.config.valid_portion
        if model_path is None:
            model_path = self.config.model_path
            self.model_path = model_path
        assert valid_portion >= 0 and valid_portion < 1.0

        if valid_path is None:
            train_stream, valid_stream = self.dataset.get_train_stream(train_path, valid_portion)
        else:
            train_stream = self.dataset.get_train_stream(train_path, 0.0)
            valid_stream = self.dataset.get_train_stream(valid_path, 0.0)

        # Build the Blocks stuff for training
        if self.m is None:
            self.m = self.config.Model(self.config, self.dataset) # with word2id
        if self.model is None:
            self.model = Model(self.m.sgd_cost) 
        #cg = ComputationGraph(self.m.weights)
        #f_weight = theano.function(cg.inputs, self.m.weights)
        #for data in train_stream.get_epoch_iterator():
        #    print(data[train_stream.sources.index("distance")].shape)
        #    weights = f_weight(data[train_stream.sources.index("distance")])
        #    print(weights)
        #    raw_input("continue")
        algorithm = GradientDescent(cost=self.m.sgd_cost,
                                    step_rule=self.config.step_rule,
                                    parameters=self.model.parameters+[self.m.delta],
                                    on_unused_sources = "ignore")
        extensions = [
        TrainingDataMonitoring(
            [v for l in self.m.monitor_vars for v in l],
            prefix='train',
            every_n_batches= self.config.print_freq)
            ]

        if self.config.save_freq is not None and model_path is not None:
            extensions += [
                SaveLoadParams(path=model_path,
                                model=self.model,
                                before_training=True,    # if exist model, the program will load it first
                                after_training=True,
                                after_epoch=True,
                                every_n_batches=self.config.save_freq)
            ]
        if valid_stream is not None and self.config.valid_freq != -1:
            extensions += [
                DataStreamMonitoring(
                    [v for l in self.m.monitor_vars_valid for v in l],
                    valid_stream,
    #                before_first_epoch = False,
                    prefix='valid',
                    every_n_batches=self.config.valid_freq),
            ]
        extensions += [
                Printing(every_n_batches=self.config.print_freq, after_epoch=True),
                ProgressBar()
        ]

        main_loop = MainLoop(
            model=self.model,
            data_stream=train_stream,
            algorithm=algorithm,    # learning algorithm: AdaDelta, Momentum or others
            extensions=extensions
        )
        # Run the model !
        main_loop.run()

    def init(self):
        self.config = WLSTMC()
        self.dataset = WLSTMD(self.config)

class BDLSTME(MTLE):
    '''
    Bi-direction LSTM Entrance: order_lstm(mention_end)||reverse_lstm(mention_begin)
    '''
    def __init__(self):
        super(BDLSTME, self).__init__()

    def init(self):
        self.config = BDLSTMC()
        self.dataset = BDLSTMD(self.config)

class BDLSTME2(MTLE):
    '''
    Bi-direction LSTM Entrance: order_lstm(mention_begin-1)||max_pooling(mention)||reverse_lstm(mention_end+1)
    '''
    def __init__(self):
        super(BDLSTME2, self).__init__()

    def init(self):
        self.config = BDLSTMC2()
        self.dataset = BDLSTMD2(self.config)

#endregion


class UTHE(object):
    '''

    '''

    def __init__(self):
        self.config = UTHC
        self.dataset = BUTHD(self.config)
        self.model = None

    def train(self, train_path = None, model_path = None):
        '''
                Train a user-text-hashtag lstm model with given training dataset or the default dataset which is defined with config.train_path

                @param train_path: path of the training dataset, file or directory, default: config.train_path
                                   File foramt: Mention TAB True_label TAB Context

                @param valid_portion: a float value define the portion of validation, default: config.multi_time_lstm.MLTC.valid_portion
                                      size of validation dataset: all_the_sample_num * valid_portion

                @param valid_path: path of the validation dataset, file or directory, if given, the valid_portion will be 0.


                @param model_path: path to dump the trained model, default: config.multi_time_lstm module.model_path
                '''
        if train_path is None:
            train_path = self.config.train_path
        if model_path is None:
            model_path = self.config.model_path
            self.model_path = model_path

        # Build train stream
        train_stream = self.dataset.get_shuffled_stream(train_path)

        # Build model

        self.model = self.config.Model(self.config, self.dataset)  # with word2id

        cg = Model(self.model.sgd_cost)

        algorithm = GradientDescent(cost=self.model.sgd_cost,
                                    step_rule=self.config.step_rule,
                                    parameters=cg.parameters,
                                    on_unused_sources='ignore')

        if plot_avail:
            extensions = [FinishAfter(after_n_epochs=1),
                    TrainingDataMonitoring([v for l in self.model.monitor_vars for v in l],
                                           prefix = 'train',
                                           every_n_batches=self.config.print_freq),
                    Plot('Training Process',
                         channels=[[self.model.monitor_vars[0][0].name], [self.model.monitor_vars[1][0].name]],
                         after_batch=True)]
        else:
            extensions = [
                TrainingDataMonitoring(
                    [v for l in self.model.monitor_vars for v in l],
                    prefix='train',
                    every_n_batches=self.config.print_freq)
            ]

        if self.config.save_freq is not None and model_path is not None:
            extensions += [
                SaveLoadParams(path=model_path,
                               model=cg,
                               before_training=True,  # if exist model, the program will load it first
                               after_training=True,
                               after_epoch=True,
                               every_n_batches=self.config.save_freq)
            ]

        extensions += [
            Printing(every_n_batches=self.config.print_freq, after_epoch=True),
            ProgressBar()
        ]

        main_loop = MainLoop(
            model=cg,
            data_stream=train_stream,
            algorithm=algorithm,  # learning algorithm: AdaDelta, Momentum or others
            extensions=extensions
        )
        # Run the model !
        main_loop.run()

    def test(self, test_path = None, test_result_path = None, model_path = None):
        raise NotImplementedError('subclasses must override test()!')

    def predict(self, predict_path = None, predict_result_path = None, model_path = None):
        raise NotImplementedError('subclasses must override predict()!')

if __name__ == "__main__":
    # Test UTHE
    entrance = UTHE()
    entrance.train()
    pass