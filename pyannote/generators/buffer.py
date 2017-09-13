#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

from collections import deque
from pyannote.core import Segment
import numpy as np
import math


def bufferize(source, target, process_func=None, agg_func=None):
    """

    Parameters
    ----------
    source : (Segment, data) iterator or SlidingWindowFeature
        e.g. MFCC features
    target : Segment iterator
        e.g. sliding window to extract sequences of MFCC features
    process_func : callable, optional
        Takes a (n_samples, n_features) numpy-array as input.
        Returns a (n_samples, n_output) numpy-array.
        E.g. a seq2seq neural network
    agg_func : callable
        Takes a (n_overlap, n_samples, n_output) numpy-array as input
        Returns a (n_samples, n_output) numpy-array.
        E.g. np.nanmean(..., axis=0)

    Returns
    -------
    buffer : (Segment, data) iterator

    Example
    -------

    """

    if process_func is None:
        process_func = lambda x: x

    # initialize source buffer
    buffer_size = source.sliding_window.samples(target.duration, mode='center')
    frame_buffer = deque([], buffer_size)
    data_buffer = deque([], buffer_size)

    # consume source until buffer is filled
    while len(frame_buffer) < buffer_size:
        frame, data = next(source)
        frame_buffer.append(frame)
        data_buffer.append(data)

    if agg_func is not None:
        first_buffer = process_func(np.stack(data_buffer))
        n_overlap = math.ceil(target.duration / target.step)
        shape = (n_overlap,) + first_buffer.shape
        buffers = np.ones((n_overlap,) + first_buffer.shape,
                          dtype=first_buffer.dtype)
        buffers *= np.nan
        shift = source.sliding_window.samples(target.step, mode='center')
        pad_width = tuple((0, 0) if s != 1 else (0, shift) for s, _ in enumerate(shape))

    incomplete = False
    for i, window in enumerate(target):

        # consume source until we've had enough to cover target window
        while frame_buffer[-1].end < window.end:
            try:
                frame, data = next(source)
            except StopIteration as e:
                # source has been consumed entirely
                # and target window is not covered
                incomplete = True
                print('frame', frame_buffer[-1])
                print('window', window)
                break
            frame_buffer.append(frame)
            data_buffer.append(data)

        if incomplete:
            break

        # process (complete) buffer
        next_buffer = process_func(np.stack(data_buffer))
        # [ 0 1 2 3 ] in example below

        # yield (processed) buffer directly when no aggregation is required
        if agg_func is None:
            yield window, next_buffer
            continue

        # shift all buffers to the left and fill with NaNs
        # | 1 2 3 4 |     | 2 3 4 _ |
        # | 5 6 7 _ | ==> | 6 7 _ _ |
        # | 8 9 _ _ |     | 9 _ _ _ |
        # | 0 _ _ _ |     | _ _ _ _ |
        buffers = np.pad(buffers, pad_width, 'constant',
                         constant_values=np.NAN)[:, shift:]

        buffers[i % n_overlap] = next_buffer
        # | 2 3 4 _ |     | 2 3 4 _ |
        # | 6 7 _ _ | ==> | 6 7 _ _ |
        # | 9 _ _ _ |     | 9 _ _ _ |
        # | _ _ _ _ |     | 0 1 2 3 |

        sub_window = Segment(window.start, window.start + target.step)
        yield sub_window, agg_func(buffers[:, :shift])

    if agg_func is None:
        raise StopIteration()

    agg_buffer = agg_func(buffers[:, shift:])

    sw = SlidingWindow(start=window.start + target.step,
                       duration=target.step, step=target.step,
                       end=window.end)

    for s, sub_window in enumerate(sw):
        yield sub_window, agg_buffer[s * shift: (s+1) * shift]

