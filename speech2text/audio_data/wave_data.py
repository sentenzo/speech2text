import wave
from dataclasses import dataclass, field
from io import BytesIO
from os import PathLike

from .audio_data import IAudioData
from .pcm_params import WHISPER_PCM_PARAMS, PcmParams


@dataclass
class WavData(IAudioData):
    _pcm_params: PcmParams = WHISPER_PCM_PARAMS
    _data: bytearray = field(default_factory=bytearray)

    @staticmethod
    def load_from_wav_file(wav_file: str | bytes | PathLike) -> "WavData":
        return WavData(*WavData._load_from_wav_file(wav_file))

    def append_chunk(self, chunk: bytes | bytearray) -> None:
        self._data.extend(chunk)

    def save_as_wav_file(self, wav_file: str | bytes | PathLike) -> None:
        with wave.open(wav_file, "wb") as file:
            file.setparams(self.pcm_params.wav_params)
            file.writeframes(self._data)

    def create_io_stream(self) -> BytesIO:
        in_memory_wav_file = BytesIO()
        self.save_as_wav_file(in_memory_wav_file)
        in_memory_wav_file.seek(0)
        return in_memory_wav_file

    @property
    def pcm_params(self) -> PcmParams:
        return self._pcm_params

    @property
    def raw_data(self) -> bytes:
        return self._data

    def split_in_chunks(self, chunk_len_sec: float = 0.5):
        chunk_len_bytes = self.pcm_params.seconds_to_byte_count(chunk_len_sec)
        data_len = len(self._data)
        if data_len == 0:
            return [bytearray()]
        return [
            self._data[pos : pos + chunk_len_bytes]
            for pos in range(0, data_len, chunk_len_bytes)
        ]


if __name__ == "__main__":
    IN_FILE_PATH = "../../../tests/audio_samples/en_chunk.wav"
    wav_data = WavData.load_from_wav_file(IN_FILE_PATH)
    empty = WavData(wav_data.pcm_params)

    for chunk in wav_data.split_in_chunks():
        empty.append_chunk(chunk)

    assert wav_data.raw_data == empty.raw_data
