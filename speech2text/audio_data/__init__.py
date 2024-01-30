"""This module provides a unified interface for different encoding formats
used in `speech2text`.

All the formats are based on PCM encoding, but the samples are stored in 
diferent data structures:
- `WavData` — samples are stored as bytes
    - basically a thin abstraction over a WAVE-file descriptor
- `PdData` — samples are wrapped in `pydub.AudioSegment` a object
    - allows doing loads of fancy transformations
- `NpData` — samples are stored in a `NumPy`-array of floats
    - `np` data types are compatible with some more suffisticated tools (in
    particular: with `whisper` and `noisereduce`)

How it works in general:
1. Raw WAVE data (a file or a byte stream from microphone)
- converting to `WavData`
2. `WavData`
- (optional) concating with the previous WAVE chunk
- converting to `PdData`
3. `PdData`
- stereo to mono
- adjusting frame rate
- splitting on silence
- (optional) normalizing sound (aligning the loudness)
- (optional) amplifying human voice frequencies
- converting to `NpData`
4. `NpData`
- (optional) noise reduction
- transcribing via OpenAI Whisper
5. Transcribed text
"""


from .np_array import NpData
from .pcm_params import WHISPER_PCM_PARAMS, PcmParams
from .pydub_audioseg import PdData
from .wave_data import WavData

__all__ = [
    "IAudioData",
    "NpData",
    "PcmParams",
    "PdData",
    "WHISPER_PCM_PARAMS",
    "WavData",
]
