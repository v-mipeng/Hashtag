# -*- coding: UTF-8 -*-

import json;
import random;
import tornado;
from tornado import gen, httpserver, ioloop, web;

import sys
sys.path.append("./source")
from entrance.lstm_entrance import MTLE
print("Initialize Model...")
predictor = MTLE()
print("Model initialized!")

class post_example_handler(web.RequestHandler):
	@web.asynchronous
	@gen.coroutine
	def post(self):
		received_json = json.loads(self.request.body);
		queries = []
		id = received_json['id']
		offset = int(received_json['offset'])
		length = int(received_json['length'])
		context = received_json['text']
		mention = context[offset:offset+length]
		queries.append((mention, offset, context))
		labels, confidences = predictor.predict_tuple_samples(samples = queries)
		label = labels[0]
		confidence = confidences[0]

		self.write(json.dumps({'id': id, 'label': label, 'confidence':str(confidence)}))
		self.finish();

if __name__ == '__main__':
    application = web.Application(
		[
			("/TypeEntities", post_example_handler),
		],
		debug = True);
    http_server = httpserver.HTTPServer(application);
    http_server.listen(8999);
    tornado.ioloop.IOLoop.instance().start()