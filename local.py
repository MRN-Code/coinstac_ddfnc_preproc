import os
import json
import sys
import numpy as np
import nibabel as nib
from utils import listRecursive
import utils as ut
from .window_operations import WindowFactory, ExemplarWindowFactory

DEFAULT_window_len = 22
DEFAULT_subject_files = ['subject_01.npy', 'subject_02.npy']
DEFAULT_fnc_measure = 'correlation'


def br_local_compute_windows(args, **kwargs):
    """
        # Description:
            Compute dFNC windows prior to clustering

        # PREVIOUS PHASE:
            NA

        # INPUT:
            window_len - length of the windo
            measure - measure to use for the connectivity matrices
            subject_files - the data files to load
            exemplar - whether or not the windows are exemplars

        # OUTPUT:
            all_windows - the computed windows
            window_indices - the subject indices computed for each window

        # NEXT PHASE:
            remote_init_env
    """
    state = args["state"]
    inputs = args["input"]
    window_len = kwargs['window_len'] if 'window_len' in inputs.keys(
    ) else DEFAULT_window_len
    measure = kwargs['measure'] if 'measure' in inputs.keys(
    ) else DEFAULT_fnc_measure
    subject_files = inputs['subject_tcs']
    exemplar = kwargs['exemplar'] if 'exemplar' in kwargs.keys() else False
    computation_phase = "dfncpp_local_1"

    if exemplar:
        window_factory = ExemplarWindowFactory(window_len=window_len,
                                               measure=measure)
        computation_phase += "_exemplar"
    else:
        window_factory = WindowFactory(window_len=window_len, measure=measure)
    all_windows = []
    window_indices = []
    for i, subject_file in enumerate(subject_files):
        subject_timecourse = nib.load(subject_file).get_data()
        if i == 1:
            ut.log('Shape of a TC is %s' % (str(subject_timecourse.shape)), state)
        all_windows += window_factory.make_windows(subject_timecourse)
        window_indices += [i for w in all_windows]
    if exemplar:
        window_indices_file = os.path.join(state['outputDirectory'], 'exemplar_window_indices.npy')
        all_windows_file = os.path.join(state['outputDirectory'], 'exemplar_all_windows.npy')
    else:
        window_indices_file = os.path.join(state['outputDirectory'], 'window_indices.npy')
        all_windows_file = os.path.join(state['outputDirectory'], 'all_windows.npy')

    ut.log('Saving window indices', state)
    np.save(window_indices_file, window_indices)
    ut.log('Saving all windows', state)
    np.save(all_windows_file, all_windows)
    computation_output = dict(output=dict(
        all_windows=all_windows_file,
        window_indices=window_indices_file,
        exemplar=exemplar,
        computation_phase=computation_phase),
        state=state
    )

    return computation_output


def fuse_output(computation_outputs):
    """
        For a list of computation outputs (as dicts)
        fuse keywords and values according to the indices passed.
        into a final output dict.
    """
    fused_output = dict(output={})
    for i, computation_output in enumerate(computation_outputs):
        for k, v in computation_output:
            fused_output['output']['%s_%d' % (k, i)] = v
            if k == 'computation_phase':
                # store the last computation phase without an integer
                fused_output['output'][k] = v

    return fused_output


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, 'computation_phase'))
    if not phase_key:  # FIRST PHASE
        computation_output_exemplar = local_compute_windows(
            exemplar=True, **parsed_args['input'])
        computation_output_all = local_compute_windows(**parsed_args['input'])
        computation_output = fuse_output(computation_output_exemplar,
                                         computation_output_all)
        sys.stdout.write(computation_output)
    else:
        raise ValueError('Phase error occurred at LOCAL')
