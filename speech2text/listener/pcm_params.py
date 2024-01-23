from dataclasses import dataclass


@dataclass
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


WHISPER_PRESET = PcmParams(1, 2, 16_000)
