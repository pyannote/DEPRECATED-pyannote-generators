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
    def __init__(self, generator, batch_size=32):
        super(BaseBatchGenerator, self).__init__()
        self.generator = generator
        self.batch_size = batch_size
        self.batch_generator_ = self.iter_batches()

    def postprocess(self, batch, signature=None):
        return batch

    def process(self, fragment, signature=None, **kwargs):
        return fragment

    def _batch_new(self, signature=None):

        if signature is None:
            signature = self.generator.signature()

        if isinstance(signature, list):
            return [self._batch_new(signature=_signature)
                    for _signature in signature]

        elif isinstance(signature, tuple):
            return tuple([self._batch_new(signature=_signature)
                    for _signature in signature])

        elif isinstance(signature, dict):
            fragment_type = signature.get('type', None)

            if fragment_type is None:
                return {key: self._batch_new(signature=_signature)
                        for key, _signature in signature.items()}
            else:
                return []


    def _batch_add(self, fragment, signature=None, batch=None, **kwargs):

        if signature is None:
            signature = self.generator.signature()

        if batch is None:
            batch = self.batch_

        if isinstance(signature, (list, tuple)):
            for _fragment, _signature, _batch, in zip(fragment, signature, batch):
                self._batch_add(_fragment,
                                 signature=_signature,
                                 batch=_batch,
                                 **kwargs)

        elif isinstance(signature, dict):
            fragment_type = signature.get('type', None)

            if fragment_type is None:
                for key in signature:
                    self._batch_add(fragment[key],
                                     signature=signature[key],
                                     batch=batch[key],
                                     **kwargs)
            else:
                processed = self.process(fragment,
                                         signature=signature,
                                         **kwargs)
                batch.append(processed)

    def _batch_pack(self, signature=None, batch=None):

        if signature is None:
            signature = self.generator.signature()

        if batch is None:
            batch = self.batch_

        if isinstance(signature, list):
            return list(self._batch_pack(signature=_signature, batch=_batch)
                        for _signature, _batch in zip(signature, batch))

        elif isinstance(signature, tuple):
            return tuple(self._batch_pack(signature=_signature, batch=_batch)
                          for _signature, _batch in zip(signature, batch))

        elif isinstance(signature, dict):

            fragment_type = signature.get('type', None)

            if fragment_type is None:
                return {key: self._batch_pack(signature=signature[key],
                                               batch=batch[key])
                        for key in signature.items()}
            elif fragment_type in {PYANNOTE_SEGMENT, 'sequence', 'boolean'}:
                return np.stack(batch)
            else:
                return batch

    def _batch_signature(self, signature=None):

        if signature is None:
            signature = self.generator.signature()

        if isinstance(signature, list):
            return list(self._batch_signature(signature=_signature)
                        for _signature in signature)

        elif isinstance(signature, tuple):
            return tuple(self._batch_signature(signature=_signature)
                          for _signature in signature)

        elif isinstance(signature, dict):

            fragment_type = signature.get('type', None)

            if fragment_type is None:
                return {key: self._batch_signature(signature=signature[key])
                        for key in signature}
            return {'type': 'batch'}

    def signature(self):
        signature = self.generator.signature()
        return self._batch_signature()

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        return next(self.batch_generator_)

    def iter_batches(self):

        batch_size = 0
        self.batch_ = self._batch_new()

        for fragment in self.generator:

            self._batch_add(fragment)
            batch_size += 1

            # fixed batch size
            if self.batch_size > 0 and batch_size == self.batch_size:
                batch = self._batch_pack()
                yield self.postprocess(batch)
                self.batch_ = self._batch_new()
                batch_size = 0


class FileBasedBatchGenerator(BaseBatchGenerator):

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

        batch_size = 0
        self.batch_ = self._batch_new()

        if infinite:
            file_generator = itertools.cycle(file_generator)

        for current_file in file_generator:

            identifier = self.file_identifier(current_file)

            preprocessed_file = self.preprocess(current_file,
                                                identifier=identifier)

            for fragment in self.generator.from_file(preprocessed_file):

                self._batch_add(fragment, identifier=identifier)
                batch_size += 1

                # fixed batch size
                if self.batch_size > 0 and batch_size == self.batch_size:
                    batch = self._batch_pack()
                    yield self.postprocess(batch)
                    self.batch_ = self._batch_new()
                    batch_size = 0

            # variable batch size
            if self.batch_size < 1:
                batch = self._batch_pack()
                yield self.postprocess(batch)
                self.batch_ = self._batch_new()
                batch_size = 0
