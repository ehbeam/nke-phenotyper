#!/bin/python

import sys
sys.path.append('..')
import predict

predict.run_logreg_boot('dsm_diag', endpoint='meds_ad')