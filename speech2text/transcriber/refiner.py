from __future__ import annotations

import noisereduce
import numpy as np
from pydub import AudioSegment

from speech2text.pcm_params import WHISPER_PRESET, PcmParams
from speech2text.utils.samples import Samples

"""
noisereduce.reduce_noise
make it lowder
remove some frequancies

r1 >> r2 >> r3
"""


class Refiner:
    def __init__(self) -> None:
        # self.pcm_params: PcmParams = pcm_params
        # self.sample_datatype: SampleDataType = SampleDataType.BYTES
        self.transformations = []

    def __rshift__(self, obj: Refiner):
        ...

    def refine(self, samples: Samples) -> Samples:
        raise NotImplementedError

    # class _RData:
    #     def __init__(self, data:

    # def _reduce_noise(self, )


class IdentityRefiner:
    def refine(self, samples: Samples) -> Samples:
        return Samples(samples)
