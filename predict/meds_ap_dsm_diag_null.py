#!/bin/python

import sys
sys.path.append('..')
import predict

predict.run_logreg_null('dsm_diag', endpoint='meds_ap')