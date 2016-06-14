from pyannote.features.audio.yaafe import YaafeFrame, YaafeMFCC
from pyannote.core import Segment, Annotation, SlidingWindow
import random
import numpy as np
import scipy.stats


class MiniBatchSequenceGenerator(object):
    """

    Parameters
    ----------
    batch_size : int, optional
        Batch size. Defaults to 10.
    """
    def __init__(self, batch_size=10):
        super(MiniBatchSequenceGenerator, self).__init__()
        self.batch_size = batch_size

    def shape(self):
        raise NotImplementedError('')

    def _sequences(self, *protocol_item):
        raise NotImplementedError('')

    def __call__(self, protocol, mode='train'):
        batches = None
        while True:
            if mode == 'train':
                generator = protocol.train_iter()

            elif mode == 'dev':
                generator = protocol.dev_iter()

            elif mode == 'test':
                generator = protocol.test_iter()

            for protocol_item in generator:
                for items in self._sequences(*protocol_item):
                    if batches is None:
                        batches = [[] for item in items]
                    for i, item in enumerate(items):
                        batches[i].append(item)
                    if len(batches[0]) == self.batch_size:
                        batches = [np.stack(batch) for batch in batches]
                        yield batches
                        batches = [[] for item in items]


class RandomSequenceGenerator(MiniBatchSequenceGenerator):
    """docstring for BatchRandomSequenceGenerator"""
    def __init__(self, batch_size=10, n_samples=126, n_features=11):
        super(RandomSequenceGenerator, self).__init__(batch_size=batch_size)
        self.n_samples = n_samples
        self.n_features = n_features

    def shape(self):
        return (self.n_samples, self.n_features)

    def _sequences(self, *args):
        while True:
            x = np.random.randn(self.n_samples, self.n_features)
            y = np.random.randint(2, size=(self.n_samples, 1))
            yield x, y


class SADMixin:

    def y_true(self, n_samples, reference):
        """

        Parameters
        ---------
        n_samples : int
        reference : pyannote.core.Annotation
        """

        y = np.zeros((n_samples, 1))
        for segment, _ in reference.itertracks():
            i0, n = self.sw.segmentToRange(segment)
            y[i0:i0 + n] = 1
        return y


class YaafeSequenceGenerator(MiniBatchSequenceGenerator):
    """

    Parameters
    ----------
    yaafe : pyannote.features.audio.yaafe.YaafeFeatureExtractor
    duration : float, optional
        Defaults to 3.2 seconds.
    step : float, optional
        Defaults to 0.8 seconds.
    normalize : boolean, optional
        Defaults to True
    batch_size : int, optional
        Defaults to 100
    """

    def __init__(self, yaafe, duration=3.2, step=0.8, normalize=True, batch_size=100):
        super(YaafeSequenceGenerator, self).__init__(batch_size=batch_size)

        self.yaafe = yaafe
        self.sw = YaafeFrame(blockSize=self.yaafe.block_size,
                             stepSize=self.yaafe.step_size,
                             sampleRate=self.yaafe.sample_rate)

        self.duration = duration
        self.step = step

        self.normalize = normalize

    def shape(self):
        """Returns shape of each sequence (n_samples, n_features)"""
        n_features = self.yaafe.dimension()
        n_samples = self.sw.durationToSamples(self.duration)
        return (n_samples, n_features)

    def _sequences(self, wav, uem, reference):

        n_samples, n_features = self.shape()

        # extract Yaafe features as numpy array
        X = self.yaafe(wav).data

        # extract reference as numpy array
        Y = self.y_true(X.shape[0], reference)

        # only processed annotated segments
        for segment in uem:

            window = SlidingWindow(duration=self.duration, step=self.step,
                                   start=segment.start, end=segment.end)

            for subsegment in window:

                if subsegment.duration < self.duration:
                    continue

                i0, n = self.sw.segmentToRange(subsegment)
                x, y = X[i0:i0 + n], Y[i0:i0 + n]
                if x.shape != (n_samples, n_features):
                    continue

                if self.normalize:
                    x = scipy.stats.zscore(x)

                yield x, y


# class BatchSequencePairsGenerator(MiniBatchSequenceGenerator):
#
#     def __init__(self, batch_size=10, duration=2.0, normalize=True, samples_per_label=40):
#         super(BatchSequencePairsGenerator, self).__init__(batch_size=batch_size)
#         self.duration = duration
#         self.normalize = normalize
#         self.samples_per_label = samples_per_label
#
#         self.mfcc_ = YaafeMFCC(e=False, De=True, DDe=True,
#                                coefs=11, D=True, DD=True)
#         self.frame_ = YaafeFrame(blockSize=self.mfcc_.block_size,
#                                  stepSize=self.mfcc_.step_size,
#                                  sampleRate=self.mfcc_.sample_rate)
#
#     def shape(self):
#
#         n_features = 0
#         n_features += self.mfcc_.e
#         n_features += self.mfcc_.De
#         n_features += self.mfcc_.DDe
#         n_features += self.mfcc_.coefs
#         n_features += self.mfcc_.coefs * self.mfcc_.D
#         n_features += self.mfcc_.coefs * self.mfcc_.DD
#
#         n_samples = int(self.duration / self.frame_.step) + 1
#
#         return (n_samples, n_features)
#
#     def _frame_slice(self, segment):
#         i0 = int(np.rint((segment.start - self.frame_.start - .5 * self.frame_.duration) / self.frame_.step))
#         n = int(segment.duration / self.frame_.step) + 1
#         return i0, n
#
#     def _subsegment(self, segment):
#         t = random.random() * (segment.duration - self.duration)
#         return Segment(t, t + self.duration)
#
#     def _generate_segments(self, annotation, shuffle=True, repeat=True):
#
#         segments = [segment for segment, _ in annotation.itertracks()]
#         while True:
#             if shuffle:
#                 random.shuffle(segments)
#
#             for segment in segments:
#                 yield self._subsegment(segment)
#
#             if not repeat:
#                 break
#
#     def _sequences(self, wav, uem, reference):
#
#         X = self.mfcc_.extract(wav).data
#
#         # start by removing segments shorter than duration
#         annotation = Annotation()
#         for segment, track, label in reference.itertracks(label=True):
#             if segment.duration <= self.duration:
#                 continue
#             annotation[segment, track] = label
#
#         labels = list(annotation.labels())
#         random.shuffle(labels)
#
#         n_samples, n_features = self.shape()
#
#         for label in labels:
#
#             positives = self._generate_segments(annotation.subset([label]))
#             negatives = self._generate_segments(annotation.subset([label], invert=True))
#
#             anchor = next(positives)
#             i0, n = self._frame_slice(anchor)
#             x_anchor = X[i0:i0 + n]
#
#             if self.normalize:
#                 x_anchor = scipy.stats.zscore(x_anchor)
#                 if x_anchor.shape != (n_samples, n_features):
#                     continue
#
#             for i in range(self.samples_per_label):
#
#                 positive = next(positives)
#                 i0, n = self._frame_slice(positive)
#                 x_positive = X[i0:i0 + n]
#                 if x_positive.shape != (n_samples, n_features):
#                     continue
#
#                 negative = next(negatives)
#                 i0, n = self._frame_slice(negative)
#                 x_negative = X[i0:i0 + n]
#                 if x_negative.shape != (n_samples, n_features):
#                     continue
#
#                 if self.normalize:
#                     x_positive = scipy.stats.zscore(x_positive)
#                     x_negative = scipy.stats.zscore(x_negative)
#                 yield x_anchor, x_positive, [1]
#                 yield x_anchor, x_negative, [0]
#
#
# class BatchSequenceTripletsGenerator(MiniBatchSequenceGenerator):
#
#     def __init__(self, batch_size=10, duration=2.0, normalize=True, samples_per_label=40):
#         super(BatchSequenceTripletsGenerator, self).__init__(batch_size=batch_size)
#         self.duration = duration
#         self.normalize = normalize
#         self.samples_per_label = samples_per_label
#
#         self.mfcc_ = YaafeMFCC(e=False, De=True, DDe=True,
#                                coefs=4, D=True, DD=True)
#         self.frame_ = YaafeFrame(blockSize=self.mfcc_.block_size,
#                                  stepSize=self.mfcc_.step_size,
#                                  sampleRate=self.mfcc_.sample_rate)
#
#     def shape(self):
#
#         n_features = 0
#         n_features += self.mfcc_.e
#         n_features += self.mfcc_.De
#         n_features += self.mfcc_.DDe
#         n_features += self.mfcc_.coefs
#         n_features += self.mfcc_.coefs * self.mfcc_.D
#         n_features += self.mfcc_.coefs * self.mfcc_.DD
#
#         n_samples = int(self.duration / self.frame_.step) + 1
#
#         return (n_samples, n_features)
#
#     def _frame_slice(self, segment):
#         i0 = int(np.rint((segment.start - self.frame_.start - .5 * self.frame_.duration) / self.frame_.step))
#         n = int(segment.duration / self.frame_.step) + 1
#         return i0, n
#
#     def _subsegment(self, segment):
#         t = random.random() * (segment.duration - self.duration)
#         return Segment(t, t + self.duration)
#
#     def _generate_segments(self, annotation, shuffle=True, repeat=True):
#
#         segments = [segment for segment, _ in annotation.itertracks()]
#         while True:
#             if shuffle:
#                 random.shuffle(segments)
#
#             for segment in segments:
#                 yield self._subsegment(segment)
#
#             if not repeat:
#                 break
#
#     def _sequences(self, wav, uem, reference):
#
#         X = self.mfcc_.extract(wav).data
#
#         # start by removing segments shorter than duration
#         annotation = Annotation()
#         for segment, track, label in reference.itertracks(label=True):
#             if segment.duration <= self.duration:
#                 continue
#             annotation[segment, track] = label
#
#         labels = list(annotation.labels())
#         random.shuffle(labels)
#
#         n_samples, n_features = self.shape()
#
#         for label in labels:
#
#             positives = self._generate_segments(annotation.subset([label]))
#             negatives = self._generate_segments(annotation.subset([label], invert=True))
#
#             anchor = next(positives)
#             i0, n = self._frame_slice(anchor)
#             x_anchor = X[i0:i0 + n]
#
#             if x_anchor.shape != (n_samples, n_features):
#                 continue
#
#             if self.normalize:
#                 x_anchor = scipy.stats.zscore(x_anchor)
#
#             if np.any(np.isnan(x_anchor)):
#                 continue
#
#             for i in range(self.samples_per_label):
#
#                 positive = next(positives)
#                 i0, n = self._frame_slice(positive)
#                 x_positive = X[i0:i0 + n]
#                 if x_positive.shape != (n_samples, n_features):
#                     continue
#
#                 negative = next(negatives)
#                 i0, n = self._frame_slice(negative)
#                 x_negative = X[i0:i0 + n]
#                 if x_negative.shape != (n_samples, n_features):
#                     continue
#
#                 if self.normalize:
#                     x_positive = scipy.stats.zscore(x_positive)
#                     x_negative = scipy.stats.zscore(x_negative)
#
#                 if np.any(np.isnan(x_positive)):
#                     continue
#                 if np.any(np.isnan(x_negative)):
#                     continue
#
#                 yield x_anchor, x_positive, x_negative
