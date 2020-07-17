#!/bin/python

import sys
sys.path.append('..')
import predict

predict.run_logreg_boot('rdoc', endpoint='meds_ap')