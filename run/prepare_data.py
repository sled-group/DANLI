import os
import sys
from sacred import Experiment
from attrdict import AttrDict

import dataset.preprocess_gamefiles as preprocess_data
import dataset.process_edh_tfd as encode_data
from config import data_ingredient

# core_mask_op = "taskset -pc %s %d" % ('0-40', os.getpid())
# os.system(core_mask_op)

ex = Experiment('prepare_data', ingredients=[data_ingredient])


@ex.automain
def main(data_args):
    data_args = AttrDict(**data_args)
    for job in data_args.data_processing_jobs:
        if job == 'preprocess':
            preprocess_data.main(data_args)
        elif job == 'encode_ori_baseline':
            data_args.encode_original_baseline = True
            data_args.encode_with_intention = False
            encode_data.main(data_args)
        elif job == 'encode_without_intent':
            data_args.encode_original_baseline = False
            data_args.encode_with_intention = False
            encode_data.main(data_args)
        elif job == 'encode_with_intent':
            data_args.encode_original_baseline = False
            data_args.encode_with_intention = True
            encode_data.main(data_args)
        else:
            assert ValueError('%s is not a valid data processing job!' % job)
