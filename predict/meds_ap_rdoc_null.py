#!/bin/python

import sys
sys.path.append('..')
import predict

predict.run_logreg_null('rdoc', endpoint='meds_ap')