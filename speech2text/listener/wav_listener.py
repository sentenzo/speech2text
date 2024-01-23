from multiprocessing import Queue

from pydub import AudioSegment

from speech2text.utils import TickSynchronizer

from .listener import Listener
from .pcm_params import PcmParams

MSEC_IN_SEC = 1000


def _wav_recorder_proc(
    queue: Queue,
    pcm_params: PcmParams,
    chunk_size_sec: float,
    path_to_wave_file: str,
) -> None:
    chunk_size_frames = pcm_params.seconds_to_sample_count(chunk_size_sec)

    wav_file: AudioSegment = (
        AudioSegment.from_wav(path_to_wave_file)
        .set_channels(pcm_params.channels_count)
        .set_frame_rate(pcm_params.sample_rate)
    )
    samples = wav_file.get_array_of_samples()
    position = 0
    audio_is_playing = True
    try:
        with TickSynchronizer(chunk_size_sec) as ticker:
            while audio_is_playing:
                chunk = samples[position : position + chunk_size_frames]
                position += chunk_size_frames
                if len(chunk) < chunk_size_frames:
                    audio_is_playing = False
                    added_silence_frames = chunk_size_frames - len(chunk)
                    added_silence_sec = (
                        added_silence_frames / chunk_size_frames
                    ) * chunk_size_sec
                    added_silence_msec = int(MSEC_IN_SEC * added_silence_sec)
                    silence_samples = AudioSegment.silent(
                        added_silence_msec,
                        pcm_params.sample_rate,
                    ).get_array_of_samples()
                    chunk.extend(silence_samples)
                ticker.tick()
                queue.put(chunk)

            silence_samples = AudioSegment.silent(
                chunk_size_sec * MSEC_IN_SEC,
                pcm_params.sample_rate,
            ).get_array_of_samples()

            while True:
                chunk = silence_samples[:]
                ticker.tick()
                queue.put(chunk)
    except KeyboardInterrupt:
        pass


class WavFileListener(Listener):
    def __init__(self, path_to_file: str) -> None:
        super().__init__()
        self.path_to_wave_file = path_to_file
        self._recorder_proc_func = _wav_recorder_proc

    def _get_recorder_proc_args(self):
        pcm_params = self.pcm_params
        chunk_size_sec = self.chunk_size_sec
        path_to_wave_file = self.path_to_wave_file
        return (pcm_params, chunk_size_sec, path_to_wave_file)
