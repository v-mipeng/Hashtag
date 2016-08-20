# -*- coding: UTF-8 -*-

import json;
import random;
import tornado;
from tornado import gen, httpserver, ioloop, web;

import sys
sys.path.append("./source")
from entrance.lstm_entrance import MTLE, BDLSTME
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
		mention = received_json['mention']
		context = received_json['context']
		if 'offset' in received_json:
		    offset  = int(received_json['offset'])
		    queries.append((mention, offset, context))
		else:
		    queries.append((mention, context))
		labels, confidences = predictor.predict_by_tuples(samples = queries)
		label = labels[0]
		confidence = confidences[0]
		self.write(json.dumps({'id': id, 'label': label, 'confidence': str(confidence)}))
		self.finish();

if __name__ == '__main__':
    application = web.Application(
		[
			("/TypeEntities", post_example_handler),
		],
		debug = True);
    http_server = httpserver.HTTPServer(application);
    http_server.listen(8997);
    tornado.ioloop.IOLoop.instance().start()
