#!/bin/python

import sys
sys.path.append('..')
import predict

predict.run_logreg_boot('combo', endpoint='suic')