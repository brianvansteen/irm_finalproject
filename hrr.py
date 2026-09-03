"""
Holographic Reduced Representations (HRR) -- vendored third-party class.

PROVENANCE. This file is the HRR class from the ecphory/hrr implementation by
Mary Kelly and Eilene Tomkins-Flanagan, which implements Tony Plate's (1995)
Holographic Reduced Representations. It is vendored rather than installed as a
dependency, and it is kept whole and unmodified so that what was adapted is
exactly what upstream provides.

    Repository: https://github.com/ecphory/hrr

WHERE THE BOUNDARY IS. Nothing in this file is this project's own work. Every
use of it is: criteria_builder.py constructs the six frozen criterion vectors
of the Hopfield head as

    xi_k = unit( name_k  +  sum_j  name_k (*) term_j )

where (*) is the circular-convolution binding provided by HRR.__mul__, and
verifies them by unbinding with HRR.__truediv__. The construction, the
criterion definitions, the separation assertion and the unbinding audit are
in criteria_builder.py; the algebra is here.

Some methods below are not exercised by this project. They are retained
deliberately: a partially stripped copy would no longer be the implementation
being cited, and the point of vendoring is that a reader can see precisely
what was taken.

References:
    Plate, T.A. (1995). Holographic reduced representations. IEEE Transactions
        on Neural Networks, 6(3), 623-641.
    Kelly, M.A., Mewhort, D.J.K., & West, R.L. (2020). Holographic Declarative
        Memory and the fan effect. Cognitive Science, 44(11), e12904.
"""

import numpy as np
from numpy.fft import fft, ifft
import math
from collections.abc import Sequence


# ============================================================================
# HRR CLASS (from ecphory/hrr by Mary Kelly & Eilene Tomkins-Flanagan)
# ============================================================================

def invPerm(perm):
    """Find the inverse permutation given a permutation."""
    inv = np.arange(perm.size)
    inv[perm] = np.arange(perm.size)
    return inv


class HRR(Sequence):
    """Tony Plate's Holographic Reduced Representations.

    Generate a vector of values sampled from a normal distribution
    with a mean of zero and a standard deviation of 1/N.
    """

    def __init__(self, N=None, data=None, zero=False, large=0.8, small=0.2):
        self.large = large
        self.small = small
        if data is not None:
            if isinstance(data, HRR):
                self.v = data.v
            else:
                self.v = np.array(data, dtype=float)
        elif zero and N is not None:
            self.v = np.zeros(N)
        elif N is not None:
            sd = 1.0 / math.sqrt(N)
            self.v = np.random.normal(scale=sd, size=N)
            self.v /= np.linalg.norm(self.v)
        else:
            raise Exception("Must specify size or data for HRR")

        self.scale = np.linalg.norm(self.v)

    def __mul__(self, other):
        """Binding via circular convolution."""
        if isinstance(other, HRR):
            return HRR(data=ifft(fft(self.v) * fft(other.v)).real)
        else:
            return HRR(data=self.v * other)

    def __rmul__(self, other):
        return self * other

    def __pow__(self, exponent):
        """Fractional binding via exponentiation in the frequency domain."""
        x = ifft(fft(self.v) ** exponent).real
        return HRR(data=x)

    def __add__(self, other):
        """Superposition via addition."""
        if isinstance(other, HRR):
            return other + self.v
        else:
            return HRR(data=other + self.v)

    def __neg__(self):
        return HRR(data=-self.v)

    def __sub__(self, other):
        return HRR(data=self.v - other.v)

    def __invert__(self):
        """Approximate inverse for unbinding."""
        return HRR(data=self.v[np.r_[0, self.v.size - 1 : 0 : -1]])

    def __truediv__(self, other):
        """Unbinding: binding with the inverse of the cue."""
        if isinstance(other, HRR):
            return self * ~other
        else:
            return HRR(data=self.v / other)

    def __len__(self):
        return self.v.size

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.v[i]
        else:
            return HRR(data=self.v[i])

    def __setitem__(self, key, value):
        self.v[key] = value

    def magnitude(self):
        """Euclidean length of the vector."""
        return math.sqrt(self.v @ self.v)

    def __eq__(self, other):
        """Cosine similarity between two HRR vectors."""
        scale = self.scale * other.scale
        if scale == 0:
            return 0
        return (self.v @ other.v) / scale

    def unit(self):
        """Normalise to unit vector."""
        return HRR(data=self.v / self.scale)

    def size(self):
        return self.v.size

    def shape(self):
        return self.v.shape

    def __matmul__(self, other):
        if isinstance(other, HRR):
            return self.v @ other.v
        elif isinstance(other, np.ndarray) and len(other.shape) == 2:
            return HRR(data=self.v @ other)
        else:
            return self.v @ other

    def proj(self, other):
        """Projection of self onto other."""
        return (self @ other.unit()).item()

    def reject(self, other):
        """Self 'without' other (PSI negation)."""
        return self - (self.proj(other)) * other

    def __or__(self, other):
        """Variant addition operator based on magnitude thresholds."""
        if self.scale > self.large:
            return self
        elif self.scale < self.small:
            return other
        else:
            return self + other
