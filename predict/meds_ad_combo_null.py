#!/bin/python

import sys
sys.path.append('..')
import predict

predict.run_logreg_null('combo', endpoint='meds_ad')