from speech2text.transcriber.state import (
    TranscriptionStateTransformation as Transformation,
)

from .dummy import Dummy
from .pydub_refine import (
    PbIfLatency_ForceSplit,
    PdIfLatency_SpeedUp,
    PdNormalize,
    PdSplitSilence,
)

DEFAULT_PIPELINE = (
    PdNormalize()
    >> PdSplitSilence()
    >> PbIfLatency_ForceSplit()
    >> PdIfLatency_SpeedUp()
)

__all__ = ["Dummy", "Transformation", "DEFAULT_PIPELINE"]
