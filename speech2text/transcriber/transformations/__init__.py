from speech2text.transcriber.state import (
    TranscriptionStateTransformation as Transformation,
)

from .dummy import Dummy
from .nr_refine import NoiseReduce
from .pydub_refine import (
    PbIfLatency_ForceSplit,
    PdIfLatency_SpeedUp,
    PdNormalize,
    PdSaveWav,
    PdSplitSilence,
)
from .transcribe import Finalize, Transcribe

DEFAULT_PIPELINE = (
    PdNormalize()
    >> PdSplitSilence()
    # >> PbIfLatency_ForceSplit()
    # >> PdIfLatency_SpeedUp()
    # >> NoiseReduce()
    >> Transcribe()
    >> Finalize()
    # >> PdSaveWav("tests/audio_samples/out.wav")
)

__all__ = ["Dummy", "Transformation", "DEFAULT_PIPELINE"]
