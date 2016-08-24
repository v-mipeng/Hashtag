import sys
sys.path.append("..")

from blocks.extensions import Printing, SimpleExtension, FinishAfter, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent

from dataset.hashtag_dataset import  BUTHD
from paramsaveload import SaveLoadParams

from config.hashtag_config import UTHC

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

        # Prepare dataset
        self.dataset.prepare(train_path)

        # Build model
        self.model = self.config.Model(self.config, self.dataset)  # with word2id


        cg = Model(self.model.sgd_cost)

        for param in cg.shared_variables:
            print(param)
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

        for date_offset in range(self.config.duration, self.dataset.date_span):
            train_stream = self.dataset.get_shuffled_stream(reference_date= self.dataset.FIRST_DAY,
                                                            date_offset = date_offset-1,
                                                            duration = self.config.duration)
            valid_stream = self.dataset.get_shuffled_stream(reference_date= self.dataset.FIRST_DAY,
                                                            date_offset = date_offset,
                                                            duration = self.config.duration)
            if self.config.valid_freq != -1:
                extensions += [
                    DataStreamMonitoring(
                        [v for l in self.model.monitor_vars_valid for v in l],
                        valid_stream,
                        before_first_epoch=True,
                        prefix='valid',
                        every_n_batches=self.config.valid_freq),
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