from pyannote.core import Segment
from pyannote.core import Timeline
from pyannote.core import Annotation
from pyannote.core import SlidingWindow
import random


class SlidingSegments(object):
    """Fixed-duration running segment generator

    Parameters
    ----------
    duration: float, optional
        Segment duration. Defaults to 3.2 seconds.
    step: float, optional
        Step duration. Defaults to 0.8 seconds.
    """

    def __init__(self, duration=3.2, step=0.8):
        super(SlidingSegments, self).__init__()

        if not duration > 0:
            raise ValueError('Duration must be strictly positive.')
        self.duration = duration

        if not step > 0:
            raise ValueError('Step must be strictly positive.')
        self.step = step

    def iter_segments(self, source):
        """
        Parameters
        ----------
        source : float, Segment, Timeline or Annotation
            If `float`, yield running segments within [0, source).
            If `Segment`, yield running segments within this segment.
            If `Timeline`, yield running segments within this timeline.
            If `Annotation`, yield running segments within its timeline.
        """

        if isinstance(source, Annotation):
            segments = source.get_timeline()

        elif isinstance(source, Timeline):
            segments = source

        elif isinstance(source, Segment):
            segments = [source]

        elif isinstance(source, (int, float)):
            if not self.duration > 0:
                raise ValueError('Duration must be strictly positive.')
            segments = [Segment(0, duration)]

        else:
            raise TypeError('source must be float, Segment, Timeline or Annotation')

        segments = [segment for segment in segments
                            if segment.duration > self.duration]
        if not segments:
            raise ValueError('Source must contain at least one segment longer than requested duration.')

        for segment in segments:
            window = SlidingWindow(duration=self.duration,
                                   step=self.step,
                                   start=segment.start,
                                   end=segment.end)
            for s in window:
                # this is needed because window may go beyond segment.end
                if s in segment:
                    yield s


class RandomSegments(object):
    """Random segment generator

    Parameters
    ----------
    duration: float, optional
        When provided, yield (random) subsegments with this `duration`.
        Defaults to yielding full segments.
    """
    def __init__(self, duration=0.):
        super(RandomSegments, self).__init__()
        self.duration = duration

    def pick(self, segment):
        """Pick a subsegment at random"""
        t = segment.start + random.random() * (segment.duration - self.duration)
        return Segment(t, t + self.duration)

    def iter_segments(self, source):
        """
        Parameters
        ----------
        source : float, Segment, Timeline or Annotation
            If `float`, yield random segments within [0, source).
            If `Segment`, yield random segments within this segment.
            If `Timeline`, yield random segments within this timeline.
            If `Annotation`, yield random segments within its timeline.
        """

        if isinstance(source, Annotation):
            segments = source.get_timeline()

        elif isinstance(source, Timeline):
            segments = source

        elif isinstance(source, Segment):
            segments = [source]

        elif isinstance(source, (int, float)):
            if not self.duration > 0:
                raise ValueError('Duration must be strictly positive.')
            segments = [Segment(0, duration)]

        else:
            raise TypeError('source must be float, Segment, Timeline or Annotation')

        segments = [segment for segment in segments
                            if segment.duration > self.duration]
        if not segments:
            raise ValueError('Source must contain at least one segment longer than requested duration.')

        n_segments = len(segments)
        while True:
            index = random.randrange(n_segments)
            segment = segments[index]
            if self.duration:
                if segment.duration < self.duration:
                    continue
                yield self.pick(segment)
            else:
                yield segment


class RandomTracks(object):
    """(segment, track) tuple generator"""

    def __init__(self):
        super(RandomTracks, self).__init__()

    def iter_tracks(self, from_annotation, yield_label=False):
        """Yield (segment, track) tuples

        Parameters
        ----------
        from_annotation : Annotation
            Annotation from which tracks are obtained.
        yield_label: boolean, optional
            When True, yield (segment, track, label) tuples.
            Defaults to yielding (segment, track) tuples.
        """
        segments = from_annotation.get_timeline()
        n_segments = len(segments)
        while True:
            index = random.randrange(n_segments)
            segment = segments[index]
            track = random.choice(list(from_annotation.get_tracks(segment)))
            if yield_label:
                label = from_annotation[segment, track]
                yield segment, track, label
            else:
                yield segment, track


class RandomTrackTriplets(object):
    """(anchor, positive, negative) track triplets generator

    Parameters
    ----------
    per_label: int, optional
        Number of consecutive triplets yielded with the same anchor label
        before switching to another label.
    """

    def __init__(self, per_label=40):
        super(RandomTrackTriplets, self).__init__()
        self.per_label = per_label

    def iter_triplets(self, from_annotation, yield_label=False):
        """Yield (anchor, positive, negative) triplets of tracks

        Parameters
        ----------
        from_annotation : Annotation
            Annotation from which triplets are obtained.
        yield_label: boolean, optional
            When True, yield triplets of (segment, track, label) tuples.
            Defaults to yielding triplets of (segment, track) tuples.
            Useful for logging which labels are more difficult to discriminate.
        """
        for label in from_annotation.labels():

            p = RandomTracks()
            positives = p.iter_tracks(from_annotation.subset([label]),
                                      yield_label=yield_label)

            n = RandomTracks()
            negatives = n.iter_tracks(from_annotation.subset([label], invert=True),
                                      yield_label=yield_label)

            anchor = next(positives)

            for _ in range(self.per_label):
                yield anchor, next(positives), next(negatives)


class RandomSegmentTriplets(object):
    """(anchor, positive, negative) segment triplets generator

    Parameters
    ----------
    duration: float, optional
        When provided, yield (random) subsegments with this `duration`.
        Defaults to yielding full segments.
    per_label: int, optional
        Number of consecutive triplets yielded with the same anchor label
        before switching to another label.
    """

    def __init__(self, duration=0., per_label=40):
        super(RandomSegmentTriplets, self).__init__()
        self.duration = duration
        self.per_label = per_label

    def pick(self, segment):
        """Pick a subsegment at random"""
        t = segment.start + random.random() * (segment.duration - self.duration)
        return Segment(t, t + self.duration)

    def iter_triplets(self, from_annotation, yield_label=False):
        """Yield (anchor, positive, negative) segment triplets

        Parameters
        ----------
        from_annotation : Annotation
            Annotation from which triplets are obtained.
        yield_label: boolean, optional
            When True, yield triplets of (segment, label) tuples.
            Default to yielding segment triplets.
            Useful for logging which labels are more difficult to discriminate.
        """

        t = RandomTrackTriplets(per_label=self.per_label)

        annotation = Annotation(uri=from_annotation.uri,
                                modality=from_annotation.modality)
        for segment, track, label in from_annotation.itertracks(label=True):
            if segment.duration < self.duration:
                continue
            annotation[segment, track] = label

        if len(annotation.labels()) < 2:
            raise ValueError('Annotation must contain at least two labels with segments longer than requested duration.')

        triplets = t.iter_triplets(annotation, yield_label=yield_label)

        for triplet in triplets:

            a, p, n = [item[0] for item in triplet]

            if self.duration:
                a, p, n = [self.pick(s) for s in (a, p, n)]

            if yield_label:
                a_, p_, n_ = [item[2] for item in triplet]
                yield (a, a_), (p, p_), (n, n_)
            else:
                yield a, p, n


class RandomSegmentPairs(object):
    """((query, returned), relevance) generator

    where `query` and `returned` are segments and `relevance` is boolean
    indicates whether `returned` has the same label as `query`.

    Parameters
    ----------
    duration: float, optional
        When provided, yield (random) subsegments with this `duration`.
        Defaults to yielding full segments.
    per_label: int, optional
        Number of consecutive relevant and irrelevant pairs yielded with the
        same query label before switching to another label.

    """
    def __init__(self, duration=0., per_label=40):
        super(RandomSegmentPairs, self).__init__()
        self.duration = duration
        self.per_label = per_label

    def iter_pairs(self, from_annotation, yield_label=False):
        """Yield ((query, returned), relevance)

        Parameters
        ----------
        from_annotation : Annotation
            Annotation from which triplets are obtained.
        yield_label: boolean, optional
            When True, yield triplets of (segment, label) tuples.
            Default to yielding segment triplets.
            Useful for logging which labels are more difficult to discriminate.
        """

        t = RandomSegmentTriplets(duration=self.duration,
                                  per_label=self.per_label)
        triplets = t.iter_triplets(from_annotation, yield_label=yield_label)

        for query, positive, negative in triplets:
            yield (query, positive), True
            yield (query, negative), False
