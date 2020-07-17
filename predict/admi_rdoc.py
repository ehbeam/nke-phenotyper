#!/bin/python

import sys
sys.path.append('..')
import predict

predict.run_logreg('rdoc', endpoint='admi')