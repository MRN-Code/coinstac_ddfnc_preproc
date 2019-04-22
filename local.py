import json
import sys
import numpy as np
from ancillary import list_recursive
from window_operations import WindowFactory, ExemplarWindowFactory

DEFAULT_window_len = 44
DEFAULT_subject_files = ['subject_01.npy', 'subject_02.npy']
DEFAULT_fnc_measure = 'correlation'


def br_local_compute_windows(**kwargs):
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

    window_len = kwargs['window_len'] if 'window_len' in kwargs.keys() else DEFAULT_window_len
    measure = kwargs['measure'] if 'measure' in kwargs.keys() else DEFAULT_fnc_measure
    subject_files = kwargs['subject_files'] if 'subject_files' in kwargs.keys() else DEFAULT_subject_files
    exemplar = kwargs['exemplar'] if 'exemplar' in kwargs.keys() else False
    computation_phase = "dfncpp_local_1"

    if exemplar:
        window_factory = ExemplarWindowFactory(window_len=window_len, measure=measure)
        computation_phase += "_exemplar"
    else:
        window_factory = WindowFactory(window_len=window_len, measure=measure)
    all_windows = []
    windows_indices = []
    for i, subject_file in enumerate(subject_files):
        subject_timecourse = np.load(subject_file)
        all_windows += window_factory.make_windows(subject_timecourse)
        window_indices += [i for w in all_windows]

    computation_output = dict(
        output=dict(
            all_windows=all_windows,
            window_indices=window_indices,
            computation_phase=computation_phase
        ),
    )

    return json.dumps(computation_output)


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
        computation_output_exemplar = local_compute_windows(exemplar=True, **parsed_args['input'])
        computation_output_all = local_compute_windows(**parsed_args['input'])
        computation_output = fuse_output(computation_output_exemplar, computation_output_all)
        sys.stdout.write(computation_output)
    else:
        raise ValueError('Phase error occurred at LOCAL')
