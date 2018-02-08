#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2018 CNRS

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


import warnings
import numpy as np
import random
from pyannote.database.util import get_unique_identifier
from .background import BackgroundGenerator


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class EndOfBatch(metaclass=Singleton):
    pass


class InputOutputSignatureMismatch(Exception):
    pass


def batchify(generator, signature, batch_size=32,
             incomplete=False, prefetch=0):
    """Pack and yield batches out of a generator

    Parameters
    ----------
    generator : iterable
        Generator
    signature : dict, optional
        Signature of the generator.
    batch_size : int, optional
        Batch size. Defaults to 32.
    incomplete : boolean, optional
        Set to True to yield final batch, even if it is incomplete (i.e.
        smaller than requested batch size). Default behavior is to not
        yield incomplete final batch.
    prefetch : int, optional
        Prefetch that many batches in a background thread.
        Defaults to not prefetch anything.


    Returns
    -------
    batch_generator : iterable
        Batch generator
    """

    class Generator(object):

        def __iter__(self):
            return self

        def next(self):
            return self.__next__()

        def __next__(self):
            return next(generator)

        def signature(self):
            return signature

    batches = BaseBatchGenerator(
        Generator(), batch_size=batch_size, incomplete=incomplete)

    if prefetch:
        batches = BackgroundGenerator(batches, max_prefetch=prefetch)

    for batch in batches:
        yield batch


class BaseBatchGenerator(object):
    """Base class to pack and yield batches out of a generator

    Parameters
    ----------
    generator : iterable
        Internal generator from which batches are packed
    batch_size : int, optional
        Defaults to 32.
    incomplete : boolean, optional
        Set to True to yield final batch, even if it is incomplete (i.e.
        smaller than requested batch size). Default behavior is to not
        yield incomplete final batch.
    """
    def __init__(self, generator, batch_size=32, incomplete=False):
        super(BaseBatchGenerator, self).__init__()
        self.generator = generator
        self.batch_size = batch_size
        self.incomplete = incomplete
        self.batch_generator_ = self.iter_batches()

        signature_in = self.generator.signature()
        signature_out = self.signature()
        self._check_signature(signature_in, signature_out)

    def _passthrough(self, x, **kwargs):
        return x

    # signature

    def _check_signature(self, sign_in, sign_out):

        if type(sign_in) == dict:

            if 'type' in sign_in:
                return True

            if sign_in.keys() != sign_out.keys():
                msg = 'Key mismatch (input: {keys_in}, output: {keys_out}).'
                raise InputOutputSignatureMismatch(
                    msg.format(keys_in=sign_in.keys(), keys_out=sign_out.keys()))

            for key in sign_out:
                self._check_signature(sign_in[key], sign_out[key])

        elif type(sign_in) != type(sign_out):

            msg = 'Type mismatch (input: {type_in}, output: {type_out}).'
            raise InputOutputSignatureMismatch(
                msg.format(type_in=type(sign_in),
                           type_out=type(sign_out)))

        elif len(sign_in) != len(sign_out):

            msg = 'Length mismatch (input: {length_in}, output: {length_out}).'
            raise InputOutputSignatureMismatch(
                msg.format(length_in=len(sign_in), length_out=len(sign_out)))

        else:

            for si, so in zip(sign_in, sign_out):
                self._check_signature(si, so)


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
                                           self._passthrough)
                    processed = process_func(fragment, signature=signature_in,
                                             **kwargs)
                    self._batch_add(processed, None, signature_out, batch=batch)

    def pack_ndarray(self, ndarrays):
        return np.stack(ndarrays)

    def pack_batch(self, batches):
        try:
            return np.stack(batches)
        except ValueError as e:
            return np.vstack(batches)

    def pack_scalar(self, scalars):
        return np.array(scalars)

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
                        for key in signature_out}
            else:
                pack_func = getattr(self, 'pack_' + fragment_type,
                                    self._passthrough)
                return pack_func(batch)

    def _postprocess(self, batch, signature):

        if type(signature) == list:
            return [self._postprocess(_batch, _signature)
                    for _batch, _signature in zip(batch, signature)]

        elif type(signature) == tuple:
            return tuple([self._postprocess(_batch, _signature)
                         for _batch, _signature in zip(batch, signature)])

        elif type(signature) == dict:
            batch_type = signature.get('type', None)
            if batch_type is None:
                return {key: self._postprocess(batch[key], signature[key])
                        for key in signature}

            else:
                postprocess_func = getattr(self, 'postprocess_' + batch_type,
                                           self._passthrough)
                return postprocess_func(batch)

    def postprocess(self, batch):
        return self._postprocess(batch, self.signature())

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        return next(self.batch_generator_)

    def iter_batches(self):

        endOfBatch = EndOfBatch()

        signature_in = self.generator.signature()
        signature_out = self.signature()

        batch_size = 0
        complete = False

        self.batch_ = self._batch_new(signature_out)

        for fragment in self.generator:

            if fragment is endOfBatch:
                complete = True
            else:
                self._batch_add(fragment, signature_in, signature_out)
                batch_size += 1

            complete |= self.batch_size > 0 and batch_size == self.batch_size

            if complete:
                if batch_size:
                    batch = self._batch_pack(signature_out)
                    yield self.postprocess(batch)
                self.batch_ = self._batch_new(signature_out)
                batch_size = 0
                complete = False

        # yield last incomplete batch
        if batch_size > 0 and self.incomplete:
            batch = self._batch_pack(signature_out)
            yield self.postprocess(batch)


def forever(iterable, shuffle=False):
    """Loop over the iterable indefinitely.

    Parameters
    ----------
    iterable : iterable
    shuffle : bool, optional
        Shuffle iterable after each full consumption
    """
    saved = list(iterable)
    while saved:
        if shuffle:
            random.shuffle(saved)
        for element in saved:
              yield element


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

    def from_file(self, current_file, incomplete=True):
        """

        Parameters
        ----------
        current_file :
        incomplete : boolean, optional
            Set to False to not yield final batch if its incomplete (i.e.
            smaller than requested batch size). Default behavior is to yield
            incomplete final batch.
        """
        def current_file_generator():
            yield current_file
        for batch in self.__call__(current_file_generator(),
                                   infinite=False,
                                   incomplete=incomplete):
            yield batch

    def __call__(self, file_generator, infinite=False,
                 robust=False, incomplete=False):
        """Generate batches by looping over a (possibly infinite) set of files

        Parameters
        ----------
        file_generator : iterable
            File generator yielding dictionaries at least containing the 'uri'
            key (uri = uniform resource identifier). Typically, one would use
            the 'train' method of a protocol available in pyannote.database.
        infinite : boolean, optional
            Loop over the file generator indefinitely, in random order.
            Defaults to exhaust the file generator only once, and then stop.
        robust : boolean, optional
            Set to True to skip files for which preprocessing fails.
            Default behavior is to raise an error.
        incomplete : boolean, optional
            Set to True to yield final batch, even if its incomplete (i.e.
            smaller than requested batch size). Default behavior is to not
            yield incomplete final batch. Has no effect when infinite is True.

        See also
        --------
        pyannote.database
        """

        signature_in = self.generator.signature()
        signature_out = self.signature()

        batch_size = 0
        self.batch_ = self._batch_new(signature_out)

        if infinite:
            file_generator = forever(file_generator, shuffle=True)

        for current_file in file_generator:

            uri = get_unique_identifier(current_file)

            try:
                preprocessed_file = self.preprocess(
                    current_file, identifier=uri)
            except Exception as e:
                if robust:
                    msg = 'Cannot preprocess file "{uri}".'
                    warnings.warn(msg.format(uri=uri))
                    continue
                else:
                    raise e

            for fragment in self.generator.from_file(preprocessed_file):

                self._batch_add(fragment, signature_in, signature_out, identifier=uri)
                batch_size += 1

                # fixed batch size
                if self.batch_size > 0 and batch_size == self.batch_size:
                    batch = self._batch_pack(signature_out)
                    yield self.postprocess(batch)
                    self.batch_ = self._batch_new(signature_out)
                    batch_size = 0

            # mono-batch
            if self.batch_size < 1:
                batch = self._batch_pack(signature_out)
                yield self.postprocess(batch)
                self.batch_ = self._batch_new(signature_out)
                batch_size = 0

        # yield incomplete final batch
        if batch_size > 0 and batch_size < self.batch_size and incomplete:
            batch = self._batch_pack(signature_out)
            yield self.postprocess(batch)
