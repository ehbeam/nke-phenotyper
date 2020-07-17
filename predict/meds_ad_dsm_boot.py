#!/bin/python

import sys
sys.path.append('..')
import predict

predict.run_logreg_boot('dsm', endpoint='meds_ad')