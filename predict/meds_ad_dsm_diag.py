#!/bin/python

import sys
sys.path.append('..')
import predict

predict.run_logreg('dsm_diag', endpoint='meds_ad')