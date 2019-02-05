"""
_INIT_PATHS.PY
Created on 11/12/2018 by A.Antonenka

Copyright (c) 2018, Arlo Technologies, Inc.
350 East Plumeria, San Jose California, 95134, U.S.A.
All rights reserved.

This software is the confidential and proprietary information of
Arlo Technologies, Inc. ("Confidential Information"). You shall not
disclose such Confidential Information and shall use it only in
accordance with the terms of the license agreement you entered into
with Arlo Technologies.
"""
"""
This is a single script to provide paths initiation to finetune pre-trained RFCN-DCN NN
"""
import os,sys
addPath = lambda rel_path: sys.path.insert(0, os.path.join(os.path.dirname(__file__), rel_path))

addPath('rfcn')
addPath('lib')