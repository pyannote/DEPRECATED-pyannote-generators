#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016 CNRS

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


import numpy as np
import itertools
from pyannote.core import PYANNOTE_SEGMENT
from pyannote.core import PYANNOTE_TRACK
from pyannote.core import PYANNOTE_LABEL


class BaseBatchGenerator(object):
    """Base class to pack batches out of a generator

    Parameters
    ----------
    generator : iterable
        Internal generator from which batches are packed
    batch_size : int, optional
        Defaults to 32.
    """
    def __init__(self, generator, batch_size=32):
        super(BaseBatchGenerator, self).__init__()
        self.generator = generator
        self.batch_size = batch_size
        self.batch_generator_ = self.iter_batches()

    def postprocess(self, batch):
        """Optional batch post-processing

        Defaults to do nothing
        """
        return batch

    def signature(self):
        raise NotImplementedError('')

    def _batch_new(self, signature_out):

        if type(signature_out) == list:
                return [self._batch_new(_signature_out)
                        for _signature_out in signature_out]

        elif type(signature_out) == tuple:
                return tuple([self._batch_new(_signature_out)
                             for _signature_out in signature_out])

        elif type(signature_out) == dict:
            fragment_type = signature_out.get('type', None)
            if fragment_type is None:
                return {key: self._batch_new(_signature_out)
                        for key, _signature_out in signature_out.items()}
            else:
                return []

    def fragment_passthrough(self, fragment, **kwargs):
        return fragment

    def _batch_add(self, fragment, signature_in, signature_out, batch=None, **kwargs):

        if batch is None:
            batch = self.batch_

        if signature_in is None:

            if type(signature_out) in (list, tuple):
                for f, s, b, in zip(fragment, signature_out, batch):
                    self._batch_add(f, None, s, batch=b, **kwargs)

            elif type(signature_out) == dict:
                fragment_type = signature_out.get('type', None)
                if fragment_type is None:
                    for key in signature_out:
                        f = fragment[key]
                        so = signature_out[key]
                        b = batch[key]
                        self._batch_add(f, None, so, batch=b, **kwargs)

                else:
                    batch.append(fragment)

        else:
            if type(signature_in) in (list, tuple):
                for f, si, so, b, in zip(fragment, signature_in, signature_out, batch):
                    self._batch_add(f, si, so, batch=b, **kwargs)

            elif type(signature_in) == dict:
                fragment_type = signature_in.get('type', None)
                if fragment_type is None:
                    for key in signature_out:
                        f = fragment[key]
                        si = signature_in[key]
                        so = signature_out[key]
                        b = batch[key]
                        self._batch_add(f, si, so, batch=b, **kwargs)

                else:
                    process_func = getattr(self, 'process_' + fragment_type,
                                           self.fragment_passthrough)
                    processed = process_func(fragment, signature=signature_in,
                                             **kwargs)
                    self._batch_add(processed, None, signature_out, batch=batch)

    def batch_passthrough(self, batch):
        return batch

    def pack_sequence(self, batch):
        return np.stack(batch)

    def pack_batch(self, batch):
        return np.stack(batch)

    def pack_boolean(self, batch):
        return np.array(batch)

    def _batch_pack(self, signature_out, batch=None):

        if batch is None:
            batch = self.batch_

        if type(signature_out) == list:
            return list(self._batch_pack(_signature_out, batch=_batch)
                        for _signature_out, _batch in zip(signature_out, batch))

        elif type(signature_out) == tuple:
            return tuple(self._batch_pack(_signature_out, batch=_batch)
                          for _signature_out, _batch in zip(signature_out, batch))

        elif type(signature_out) == dict:
            fragment_type = signature_out.get('type', None)

            if fragment_type is None:
                return {key: self._batch_pack(signature_out[key],
                                              batch=batch[key])
                        for key in signature_out.items()}
            else:
                pack_func = getattr(self, 'pack_' + fragment_type,
                                    self.batch_passthrough)
                return pack_func(batch)

    def _batch_signature(self, signature_in):

        if isinstance(signature_in, list):
            return list(self._batch_signature(_signature)
                        for _signature in signature_in)

        elif isinstance(signature_in, tuple):
            return tuple(self._batch_signature(_signature)
                          for _signature in signature_in)

        elif isinstance(signature_in, dict):

            fragment_type = signature_in.get('type', None)

            if fragment_type is None:
                return {key: self._batch_signature(signature_in[key])
                        for key in signature_in}

            return {'type': 'batch'}

    def signature(self):
        signature_in = self.generator.signature()
        return self._batch_signature(signature_in)

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        return next(self.batch_generator_)

    def iter_batches(self):

        signature_in = self.generator.signature()
        signature_out = self.signature()

        batch_size = 0

        self.batch_ = self._batch_new(signature_out)

        for fragment in self.generator:

            self._batch_add(fragment, signature_in, signature_out)
            batch_size += 1

            # fixed batch size
            if self.batch_size > 0 and batch_size == self.batch_size:
                batch = self._batch_pack(signature_out)
                yield self.postprocess(batch)
                self.batch_ = self._batch_new(signature_out)
                batch_size = 0


class FileBasedBatchGenerator(BaseBatchGenerator):
    """
    Parameters
    ----------
    generator :
        Must implement generator.from_file
    """

    # identifier is useful for thread-safe current_file dependent preprocessing
    def preprocess(self, current_file, identifier=None, **kwargs):
        """Returns pre-processed current_file
        (and optionally set internal state)
        """
        return current_file

    def file_identifier(self, current_file):
        # TODO - do better than that!!!
        wav, _, _ = current_file
        return hash(wav)

    def from_file(self, current_file):
        def current_file_generator():
            yield current_file
        for batch in self.__call__(current_file_generator(), infinite=False):
            yield batch

    def __call__(self, file_generator, infinite=False):

        signature_in = self.generator.signature()
        signature_out = self.signature()

        batch_size = 0
        self.batch_ = self._batch_new(signature_out)

        if infinite:
            file_generator = itertools.cycle(file_generator)

        for current_file in file_generator:

            identifier = self.file_identifier(current_file)

            preprocessed_file = self.preprocess(current_file,
                                                identifier=identifier)

            for fragment in self.generator.from_file(preprocessed_file):

                self._batch_add(fragment, signature_in, signature_out, identifier=identifier)
                batch_size += 1

                # fixed batch size
                if self.batch_size > 0 and batch_size == self.batch_size:
                    batch = self._batch_pack(signature_out)
                    yield self.postprocess(batch)
                    self.batch_ = self._batch_new(signature_out)
                    batch_size = 0

            # variable batch size
            if self.batch_size < 1:
                batch = self._batch_pack(signature_out)
                yield self.postprocess(batch)
                self.batch_ = self._batch_new(signature_out)
                batch_size = 0
