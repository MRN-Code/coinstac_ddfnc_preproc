"""
Remote file for multi-shot KMeans
"""

import os
import sys
import json
import numpy as np
from ancillary import list_recursive
from ica.ica import ica1

#CONFIG_FILE = 'config.cfg'
DEFAULT_data_file = 'data.txt'
DEFAULT_num_components = 20


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, 'computation_phase'))

    if 'local_noop' in phase_key:  # FIRST PHASE
        computation_output = remote_init_env(parsed_args)
        computation_output = remote_ica(parsed_args, json.loads(computation_output))
        sys.stdout.write(computation_output)
    else:
        raise ValueError('Oops')
