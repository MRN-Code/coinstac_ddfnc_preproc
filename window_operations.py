import numpy as np

MEASURES = dict(correlation=np.corrcoef,covariance=np.cov)

DEFAULT_window = 22
DEFAULT_transpose = False
DEFAULT_measure = 'correlation'


class WindowFactory(object):
    """
         WindowFactory
               Can create windows for subject timecourses
                 using a given measure:
                    correlation, covariance, mutual information (TODO)

         args:
              window_len -- length of the sliding window -- DEFAULT: 22
              measure    -- measure to use when computing the windows -- DEFAULT: 'correlation'
    """
    def __init__(self, window_len=DEFAULT_window, measure=DEFAULT_measure):
        self.window_len = window_len
        self.measure = MEASURES[measure] if measure in MEASURES.keys() else MEASURES[DEFAULT_measure]

    def make_windows(self, x):
        """ Using a sliding window, compute connectivity matrices """
        windows = []
        start = 0
        end = start + self.window_len
        while end <= x.shape[0]:
            xWindow = x[start:end, :]
            connectivity_mat = self.measure(xWindow.T)
            windows += [connectivity_mat]
            start += 1
            end = start+window_len
        return windows


class ExemplarWindowFactory(WindowFactory):
    """
         ExemplarWindowFactory
               Can create windows for subject exemplar timepoints
                 by creating windows using the parent method
                 and extracting exempalr timepoints corresponding to maximal
                 variance. 
    """
    def make_windows(self, x):
        """find timepoints of maximal variance, and use them to make exemplar windows
            args:
                  x - input signal
        """
        windows = super(ExemplarWindowFactory, self).make_windows()
        variance_windows = []
        start = 0
        end = start + self.window_len
        while end <= x.shape[0]:
            xWindow = x[start:end, :]
            variance_windows.append(np.var(xWindow))
            start += 1
            end = start+self.window_len
        maxima, indices = self.local_maxima(np.array(variance_windows))
        return [windows[i] for i in indices]

    def local_maxima(self, a, x=None, indices=True):
        """
        Finds local maxima in a discrete list 'a' to index an array 'x'
        if 'x' is not specified, 'a' is indexed.
        https://stackoverflow.com/questions/4624970/
                finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
        """
        asm = self.smooth(a)
        maxima = [asm[i] for i in np.where(np.array(np.r_[1, asm[1:] < asm[:-1]] &
                                       np.r_[asm[:-1] < asm[1:], 1]))[0]]
        matches = [find_nearest(a, maximum) for maximum in maxima]
        indices = [i for i in range(len(a)) if a[i] in matches]
        if indices:
            return matches, indices
        return matches

    def smooth(self, x, window_len=DEFAULT_window):
        """
        Smooths the window using np hanning

	args:
		x - the input signal
		window_len - length of the window
        """
        if x.ndim != 1:
            raise(ValueError, "smooth only accepts 1 dimension arrays.")

        if x.size < window_len:
            raise(ValueError, "Input vector needs to be bigger than window size.")

        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
        w = np.hanning(window_len)
        y = np.convolve(w/w.sum(), s, mode='valid')
        return y


