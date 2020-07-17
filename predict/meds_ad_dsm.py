#!/bin/python

import sys
sys.path.append('..')
import predict

predict.run_logreg('dsm', endpoint='meds_ad')