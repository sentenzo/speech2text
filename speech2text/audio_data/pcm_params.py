"""PCM stands for Pulse-code modulation (in WAVE files)

## Terminology:

- `channels_count` — the amount of speakers required to play the audio file,
usually equals to 1 (mono) or 2 (sterio)
- sample — a sequence of bytes, which encodes a single amplitude value
- frame — a sequence of bytes, which encodes samples for each channel
at some point in time
- `sample_width_bytes` — the size of one sample in bytes, usually equals to 2 or 4
- `frame_rate` — the amount of frames in a second, can technically be `float`,
but the Python's `wave` module stores it as `int`

## Constants:
- `WHISPER_PCM_PARAMS` — PCM-parameters, used by OpenAI Whisper lib
"""

from dataclasses import dataclass
from wave import Wave_read, Wave_write


@dataclass(frozen=True, slots=True)
class PcmParams:
    channels_count: int  # nchannels
    sample_width_bytes: int  # sampwidth
    frame_rate: int  # framerate

    def frame_size_bytes(self) -> int:
        """The size of one frame (all the channels) in bytes"""
        return self.sample_width_bytes * self.channels_count

    def seconds_to_frame_count(self, seconds: float) -> int:
        """How many frames fit in a time span (sec)?

        How many frames should I read from the buffer to get a time span
        with a duration of `seconds` seconds?
        """
        return int(seconds * self.frame_rate)

    def seconds_to_byte_count(self, seconds: float) -> int:
        """How many bytes should I read from the buffer to get a time span
        with a duration of `seconds` seconds?
        """
        return self.seconds_to_frame_count(seconds) * self.frame_size_bytes()

    def frame_count_to_seconds(self, frame_count: int) -> float:
        """I have `frame_count` frames. How much is it in seconds?"""
        return frame_count / self.frame_rate

    @property
    def wav_params(self):
        """Useful when you need to create a new WAVE file using `wave` module:
        ```
        with wave.open(new_file, "wb") as file:
            file.setparams(self.pcm_params.wav_params)
        ```
        """
        # see https://docs.python.org/3/library/wave.html#wave.Wave_read.getparams
        return (
            self.channels_count,  # nchannels
            self.sample_width_bytes,  # sampwidth
            self.frame_rate,  # framerate
            0,  # nframes - unknown
            "NONE",  # comptype
            "not compressed",  # compname
        )

    @staticmethod
    def from_wave_file(wave_file: Wave_read | Wave_write) -> "PcmParams":
        """Creates a new `PcmParams` object based on `wave` module file
        descriptor
        """
        return PcmParams(
            wave_file.getnchannels(),
            wave_file.getsampwidth(),
            wave_file.getframerate(),
        )


WHISPER_PCM_PARAMS = PcmParams(1, 2, 16_000)
