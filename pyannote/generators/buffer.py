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
import numpy as np


def bufferize(source, target, process_buffer=None):
    """

    Parameters
    ----------
    source : (Segment, data) iterator or SlidingWindowFeature
    target : Segment iterator

    Returns
    -------
    buffer : (Segment, data) iterator

    Example
    -------

    """

    if process_buffer is None:
        process_buffer = lambda x: x

    buffer_size = source.sliding_window.samples(target.duration, mode='center')
    frame_buffer = deque([], buffer_size)
    data_buffer = deque([], buffer_size)

    while len(frame_buffer) < buffer_size:
        frame, data = next(source)
        frame_buffer.append(frame)
        data_buffer.append(data)

    for window in target:

        while frame_buffer[0].start < window.start:
            frame, data = next(source)
            frame_buffer.append(frame)
            data_buffer.append(data)

        next_buffer = np.stack(data_buffer)
        yield window, process_buffer(next_buffer)


