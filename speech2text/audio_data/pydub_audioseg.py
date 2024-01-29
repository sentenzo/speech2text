from io import BytesIO
from os import PathLike

from pydub import AudioSegment, effects, silence

from .audio_data import IAudioData, PcmParams
from .pcm_params import WHISPER_PCM_PARAMS
from .wave_data import WavData


class PdData(AudioSegment, IAudioData):
    @staticmethod
    def load_from_wav_file(
        wav_file: str | bytes | PathLike | WavData,
    ) -> "PdData":
        if isinstance(wav_file, WavData):
            wav_file = wav_file.create_io_stream()
        return PdData.from_wav(wav_file)

    def create_io_stream(self) -> BytesIO:
        in_memory_wav_file = BytesIO()
        self.export(in_memory_wav_file, "wav")
        in_memory_wav_file.seek(0)
        return in_memory_wav_file

    def create_wav_data(self) -> WavData:
        return WavData(self.pcm_params, bytearray(self.raw_data))

    @property
    def pcm_params(self) -> PcmParams:
        return PcmParams(
            channels_count=self.channels,
            sample_width_bytes=self.sample_width,
            frame_rate=self.frame_rate,
        )

    def adjust_pcm_params(
        self, new_pcm_params: PcmParams = WHISPER_PCM_PARAMS
    ) -> "PdData":
        new_audio = (
            self.set_channels(new_pcm_params.channels_count)
            .set_frame_rate(new_pcm_params.frame_rate)
            .set_sample_width(new_pcm_params.sample_width_bytes)
        )
        return new_audio

    def normalize(self, headroom: float = 0.1) -> "PdData":
        return effects.normalize(self, headroom)

    def high_pass_filter(self, cutoff: float) -> "PdData":
        return effects.high_pass_filter(self, cutoff)

    def low_pass_filter(self, cutoff: float) -> "PdData":
        return effects.low_pass_filter(self, cutoff)

    def speedup(
        self,
        playback_speed: float,
        chunk_size: int = 150,
        crossfade: int = 25,
    ) -> "PdData":
        assert playback_speed >= 1.0
        if playback_speed == 1.0:
            return self
        return effects.speedup(self, playback_speed, chunk_size, crossfade)

    def split_on_silence(
        self,
        min_silence_len=1000,
        silence_thresh=-30,
        keep_silence=1000,
        seek_step=10,
    ):
        segments = silence.split_on_silence(
            self,
            min_silence_len,
            silence_thresh,
            keep_silence,
            seek_step,
        )
        return segments


if __name__ == "__main__":
    from pprint import pprint as pp

    IN_FILE_PATH = "../../../tests/audio_samples/en_chunk.wav"

    audio = PdData.load_from_wav_file(IN_FILE_PATH)
    audio.show_player()
    audio.show_specgram()
    pp(audio.dBFS)

    wav_data = WavData.load_from_wav_file(IN_FILE_PATH)
    assert audio.raw_data == wav_data.raw_data

    audio = audio.adjust_pcm_params()
    audio.show_player()
    audio.show_specgram()
    pp(audio.dBFS)
