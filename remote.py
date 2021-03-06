"""
Remote file for multi-shot KMeans
"""

import os
import sys
import json
import numpy as np
from utils import listRecursive

#CONFIG_FILE = 'config.cfg'
DEFAULT_data_file = 'data.txt'
DEFAULT_num_components = 20


def dfncpp_remote_noop(**kwargs):
    """
        # Description:
            Nooperation

        # PREVIOUS PHASE:
            NA

        # INPUT:

        # OUTPUT:

        # NEXT PHASE:
            remote_init_env
    """
    computation_output = dict(
        output=dict(computation_phase="dfncpp_remote_noop"), )

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, 'computation_phase'))

    if 'local_compute_windows_exemplar' in phase_key:  # FIRST PHASE
        computation_output = remote_noop(parsed_args,
                                         json.loads(computation_output))
        sys.stdout.write(computation_output)
    else:
        raise ValueError('Oops')
