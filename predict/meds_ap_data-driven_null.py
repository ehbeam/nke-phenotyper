#!/bin/python

import sys
sys.path.append('..')
import predict

predict.run_logreg_null('data-driven', endpoint='meds_ap')