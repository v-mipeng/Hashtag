import logging
import io
import os

import numpy

import cPickle

from blocks.extensions import SimpleExtension

logging.basicConfig(level='INFO')
logger = logging.getLogger('extensions.SaveLoadParams')

class SaveLoadParams(SimpleExtension):
	def __init__(self, path, model, **kwargs):
		super(SaveLoadParams, self).__init__(**kwargs)

		self.path = path
		self.model = model

	def do_save(self):
		if not os.path.exists(os.path.dirname(self.path)):
			os.makedirs(os.path.dirname(self.path))
		with open(self.path, 'wb') as f:
			logger.info('Saving parameters to %s...'%self.path)
			cPickle.dump(self.model.get_parameter_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

	def do_load(self):
		try:
			with open(self.path, 'rb') as f:
				logger.info('Loading parameters from %s...'%self.path)
				self.model.set_parameter_values(cPickle.load(f))
		except IOError as e:
			print("Cannot load parameters!")

	def do(self, which_callback, *args):
		if which_callback == 'before_training':
			self.do_load()
		else:
			self.do_save()

