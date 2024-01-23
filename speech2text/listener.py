from __future__ import annotations

import logging
import time
from multiprocessing import Process, Queue

import pyaudio as pa
from pydub import AudioSegment

import speech2text.config as cfg
from speech2text.tick import TickSynchronizer

logger = logging.getLogger(__name__)

QUEUE_CHECK_DELAY_SEC = 0.05
DEFAULT_CHUNK_SIZE_SEC = 0.5
BUFFER_SIZE_MULTI = 10
SAMPLE_FORMAT = {2: pa.paInt16, 4: pa.paInt32}[cfg.SAMPLE_WIDTH]


class Listener:
    def __init__(self) -> None:
        self.chunk_size_sec = None
        self.chunk_size_frames = None
        self.buffer_size_frames = None
        self.set_chunk_size_sec(DEFAULT_CHUNK_SIZE_SEC)

    def set_chunk_size_sec(self, chunk_size_sec):
        self.chunk_size_sec = chunk_size_sec
        self.chunk_size_frames = int(chunk_size_sec * cfg.SAMPLE_RATE)
        self.buffer_size_frames = self.chunk_size_frames * BUFFER_SIZE_MULTI
        return self

    def create_stream_recorder(self):
        raise NotImplementedError

    def gen_chunks(self):
        stream_recorder = self.create_stream_recorder()
        audio_chunks_queue = Queue()
        stream_recorder_proc = Process(
            target=stream_recorder,
            args=(audio_chunks_queue,),
        )
        stream_recorder_proc.start()
        while True:
            time.sleep(QUEUE_CHECK_DELAY_SEC)
            if not audio_chunks_queue.empty():
                yield audio_chunks_queue.get()


def wave_stream_recorder_proc(queue: Queue, path_to_wave_file: str) -> None:
    chunk_size_sec = DEFAULT_CHUNK_SIZE_SEC
    chunk_size_frames = int(chunk_size_sec * cfg.SAMPLE_RATE)

    wav_file: AudioSegment = (
        AudioSegment.from_wav(path_to_wave_file)
        .set_channels(1)  # to mono
        .set_frame_rate(cfg.SAMPLE_RATE)
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
                    added_silence_msec = int(1000 * added_silence_sec)
                    silence_samples = AudioSegment.silent(
                        added_silence_msec,
                        cfg.SAMPLE_RATE,
                    ).get_array_of_samples()
                    chunk.extend(silence_samples)
                ticker.tick()
                queue.put(chunk)

            silence_samples = AudioSegment.silent(
                chunk_size_sec * 1000,
                cfg.SAMPLE_RATE,
            ).get_array_of_samples()

            while True:
                chunk = silence_samples[:]
                ticker.tick()
                queue.put(chunk)
    except KeyboardInterrupt:
        pass


class WaveFileListener(Listener):
    def __init__(self, path_to_file: str) -> None:
        super().__init__()
        self.path_to_wave_file = path_to_file

    def gen_chunks(self):
        audio_chunks_queue = Queue()
        stream_recorder_proc = Process(
            target=wave_stream_recorder_proc,
            args=(audio_chunks_queue, self.path_to_wave_file),
        )
        stream_recorder_proc.start()
        logger.info("Start recording")
        try:
            while True:
                time.sleep(QUEUE_CHECK_DELAY_SEC)
                if not audio_chunks_queue.empty():
                    yield audio_chunks_queue.get()
        except KeyboardInterrupt:
            pass
        finally:
            logger.info("Stop recording")


class PyAudioWrapper(pa.PyAudio):
    def __init__(self) -> None:
        super().__init__()
        self.inside_cm = False

    def __enter__(self):
        assert not self.inside_cm
        self.inside_cm = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.inside_cm = False
        self.terminate()


def mic_stream_recorder_proc(queue: Queue, microphone_id: str) -> None:
    chunk_size_sec = DEFAULT_CHUNK_SIZE_SEC
    chunk_size_frames = int(chunk_size_sec * cfg.SAMPLE_RATE)
    buffer_size_frames = chunk_size_frames * BUFFER_SIZE_MULTI
    with PyAudioWrapper() as audio:
        stream = audio.open(
            # input_device_index=self.microphone_id,
            format=SAMPLE_FORMAT,
            channels=cfg.CHANNELS,
            rate=cfg.SAMPLE_RATE,
            input=True,
            frames_per_buffer=buffer_size_frames,
        )
        try:
            while True:
                chunk = stream.read(num_frames=chunk_size_frames)
                queue.put(chunk)
        except KeyboardInterrupt:
            pass
        finally:
            stream.close()


class MicrophonListener(Listener):
    def __init__(self, microphone_id: int = None):
        super().__init__()
        with PyAudioWrapper() as audio:
            if microphone_id is None:
                microphone_id = audio.get_default_input_device_info()["index"]

        self.microphone_id = microphone_id

    def gen_chunks(self):
        audio_chunks_queue = Queue()
        stream_recorder_proc = Process(
            target=mic_stream_recorder_proc,
            args=(audio_chunks_queue, self.microphone_id),
        )
        stream_recorder_proc.start()
        logger.info("Start recording")
        try:
            while True:
                time.sleep(QUEUE_CHECK_DELAY_SEC)
                if not audio_chunks_queue.empty():
                    yield audio_chunks_queue.get()
        except KeyboardInterrupt:
            pass
        finally:
            logger.info("Stop recording")
