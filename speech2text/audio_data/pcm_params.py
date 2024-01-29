"""
PCM stands for Pulse-code modulation.
"""
from dataclasses import dataclass
from wave import Wave_read, Wave_write


@dataclass(
    frozen=True,
    slots=True,
)
class PcmParams:
    channels_count: int  # nchannels
    sample_width_bytes: int  # sampwidth
    sample_rate: float  # framerate

    def sample_size_bytes(self) -> int:
        return self.sample_width_bytes * self.channels_count

    def seconds_to_sample_count(self, seconds: float) -> int:
        return int(seconds * self.sample_rate)

    def sample_count_to_seconds(self, sample_count: int) -> float:
        return sample_count / self.sample_rate

    @property
    def wav_params(self):
        # see https://docs.python.org/3/library/wave.html#wave.Wave_read.getparams
        return (
            self.channels_count,  # nchannels
            self.sample_width_bytes,  # sampwidth
            self.sample_rate,  # framerate
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
