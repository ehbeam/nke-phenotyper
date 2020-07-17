#!/bin/python

import sys
sys.path.append('..')
import predict

predict.run_logreg_boot('data-driven', endpoint='meds_ap')