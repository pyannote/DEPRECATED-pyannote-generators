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


class BaseBatchGenerator(object):
    def __init__(self, generator, batch_size=32):
        super(BaseBatchGenerator, self).__init__()
        self.generator = generator
        self.batch_size = batch_size

    # identifier is useful for thread-safe protocol_item dependent preprocessing
    def preprocess(self, protocol_item, identifier=None, **kwargs):
        """Returns pre-processed protocol_item
        (and optionally set internal state)
        """
        return protocol_item

    def process(self, fragment, signature=None, identifier=None):
        raise NotImplementedError()

    def postprocess(self, batch, signature=None):
        return batch

    def __batch_new(self, signature=None):

        if signature is None:
            signature = self.generator.signature()

        if isinstance(signature, list):
            return [self.__batch_new(signature=_signature)
                    for _signature in signature]

        elif isinstance(signature, tuple):
            return tuple([self.__batch_new(signature=_signature)
                    for _signature in signature])

        elif isinstance(signature, dict):

            fragment_type = signature.get('type', None)

            if fragment_type in set(['segment', 'boolean']):
                return []

            if fragment_type is None:
                return {key: self.__batch_new(signature=_signature)
                        for key, _signature in signature.items()}

            raise NotImplementedError('')

    def __batch_add(self, fragment, signature=None, batch=None, identifier=None):

        if signature is None:
            signature = self.generator.signature()

        if batch is None:
            batch = self.batch_

        if isinstance(signature, (list, tuple)):
            for _fragment, _signature, _batch, in zip(fragment, signature, batch):
                self.__batch_add(_fragment,
                                 signature=_signature,
                                 batch=_batch,
                                 identifier=identifier)

        elif isinstance(signature, dict):
            fragment_type = signature.get('type', None)

            if fragment_type == 'segment':
                processed = self.process(fragment,
                                         signature=signature,
                                         identifier=identifier)
                batch.append(processed)

            elif fragment_type == 'boolean':
                processed = self.process(fragment,
                                         signature=signature,
                                         identifier=identifier)
                batch.append(processed)

            else:
                for key in signature:
                    self.__batch_add(fragment[key],
                                     signature=signature[key],
                                     batch=batch[key],
                                     identifier=identifier)

    def __batch_pack(self, signature=None, batch=None):

        if signature is None:
            signature = self.generator.signature()

        if batch is None:
            batch = self.batch_

        if isinstance(signature, list):
            return list(self.__batch_pack(signature=_signature, batch=_batch)
                        for _signature, _batch in zip(signature, batch))

        elif isinstance(signature, tuple):
            return tuple(self.__batch_pack(signature=_signature, batch=_batch)
                          for _signature, _batch in zip(signature, batch))

        elif isinstance(signature, dict):

            fragment_type = signature.get('type', None)

            if fragment_type in set(['segment', 'boolean']):
                return np.stack(batch)

            else:
                return {key: self.__batch_pack(signature=signature[key],
                                               batch=batch[key])
                        for key in signature.items()}

    def from_protocol_item(self, protocol_item, identifier=None):
        item = self.preprocess(protocol_item, identifier=identifier)
        for fragment in self.generator.from_protocol_item(item):
            yield fragment

    def __call__(self, protocol_iter_func, infinite=False):

        batch_size = 0
        self.batch_ = self.__batch_new()

        first = True
        while first or infinite:
            first = False
            for identifier, protocol_item in enumerate(protocol_iter_func()):

                for fragment in self.from_protocol_item(protocol_item, identifier=identifier):

                    self.__batch_add(fragment, identifier=identifier)
                    batch_size += 1

                    if batch_size == self.batch_size:
                        batch = self.__batch_pack()
                        yield self.postprocess(batch)
                        self.batch_ = self.__batch_new()
                        batch_size = 0
