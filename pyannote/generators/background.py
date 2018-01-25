# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# This software uses a shared copyright model: each contributor holds copyright over
# their contributions to it. The project versioning records all such
# contribution and copyright details.
# By contributing to the this repository through pull-request, comment,
# or otherwise, the contributor releases their content to the license and
# copyright terms herein.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# shamelessly stolen (and slightly adapted) from
# https://github.com/justheuristic/prefetch_generator
# http://stackoverflow.com/questions/7323664/python-generator-pre-fetch


import threading
import sys
import queue


class BackgroundGenerator(threading.Thread):
    """Transform a generator into a background-thread generator.

    It is quite lightweight, but not entirely weightless.
    Using global variables inside generator is not recommended (may rise
    GIL and zero-out the benefit of having a background thread.)
    The ideal use case is when everything it requires is store inside it
    and everything it outputs is passed through queue.

    There's no restriction on doing weird stuff, reading/writing files,
    retrieving URLs (or whatever) whilst iterating.

    Parameters
    ----------
    generator: generator or genexp or any
        It can be used with any minibatch generator.
    max_prefetch: int, optional
        Defines, how many iterations (at most) can background generator
        keep stored at any moment of time. Whenever there's already
        `max_prefetch` batches stored in queue, the background process will
        halt until one of these batches is dequeued. `max_prefetch = 1`
        (default) is okay unless you deal with some weird file IO in your
        generator. Setting `max_prefetch` to -1 lets it store as many
        batches as it can, which will work slightly (if any) faster, but
        will require storing all batches in memory. If you use infinite
        generator with `max_prefetch = -1`, it will exceed the RAM size
        unless dequeued quickly enough.

    Usage
    -----
    >>> for batch in BackgroundGenerator(batch_generator):
    ...     do_something(batch)

    """

    def __init__(self, generator, max_prefetch=1):
        super(BackgroundGenerator, self).__init__(daemon=True)
        self.queue_ = queue.Queue(max_prefetch)
        self.generator = generator
        self.start()

    def run(self):
        for item in self.generator:
            self.queue_.put(item)
        self.queue_.put(None)

    def next(self):
        next_item = self.queue_.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class background:
    """Decorator that transforms a generator into a background-thread generator

    Usage
    -----
    >>> @background(max_prefect=1)
    >>> def batch_generator(some_param):
    ...     while True:
    ...         # do something
    ...         yield batch

    See also
    --------
    BackgroundGenerator

    """
    def __init__(self, max_prefetch=1):
        self.max_prefetch = max_prefetch

    def __call__(self, generator):
        def background_generator(*args,**kwargs):
            return BackgroundGenerator(generator(*args,**kwargs))
        return background_generator
