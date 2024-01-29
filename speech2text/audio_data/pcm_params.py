"""
PCM stands for Pulse-code modulation (in WAVE files)

Terminology:
channels count -- the amount of speakers required to play the audio file,
                  usually equals to 1 (mono) or 2 (sterio)
        sample -- a sequence of bytes, which encodes a single amplitude value
         frame -- a sequence of bytes, which encodes samples for each channel
                  at some point in time
  sample width -- the size of one sample in bytes, usually equals to 2 or 4
    frame rate -- the amount of frames in a second, can technically be `float`,
                  but the Python's `wave` module stores it as `int`
"""

from dataclasses import dataclass
from wave import Wave_read, Wave_write


@dataclass(frozen=True, slots=True)
class PcmParams:
    channels_count: int  # nchannels
    sample_width_bytes: int  # sampwidth
    frame_rate: int  # framerate

    def frame_size_bytes(self) -> int:
        return self.sample_width_bytes * self.channels_count

    def seconds_to_frame_count(self, seconds: float) -> int:
        return int(seconds * self.frame_rate)

    def seconds_to_byte_count(self, seconds: float) -> int:
        return self.seconds_to_frame_count(seconds) * self.frame_size_bytes()

    def frame_count_to_seconds(self, frame_count: int) -> float:
        return frame_count / self.frame_rate

    @property
    def wav_params(self):
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
        return PcmParams(
            wave_file.getnchannels(),
            wave_file.getsampwidth(),
            wave_file.getframerate(),
        )


WHISPER_PCM_PARAMS = PcmParams(1, 2, 16_000)
