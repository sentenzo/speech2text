from __future__ import annotations

import logging
import time
from multiprocessing import Process, Queue
from typing import Callable, Generator

import speech2text.config as cfg
from speech2text.audio_data import WHISPER_PCM_PARAMS, PcmParams
from speech2text.utils.tick import Ticker

logger = logging.getLogger(__name__)


class Listener:
    def __init__(self, pcm_params: PcmParams = WHISPER_PCM_PARAMS) -> None:
        self.pcm_params: PcmParams = pcm_params
        self._recorder_proc: Callable | None = None
        self._chunks_iterator: Generator | None = None
        self._latency_ratio: float | None = None

    @property
    def latency_ratio(self) -> float:
        return self._latency_ratio or 0.0

    def _get_recorder_proc_kwargs(self):
        raise NotImplementedError

    def get_chunks_iterator(self, *args, **kwargs):
        if not self._chunks_iterator:
            self._chunks_iterator = self._get_chunks_iterator(*args, **kwargs)
        return self._chunks_iterator

    def close_chunks_iterator(self):
        if self._chunks_iterator:
            self._chunks_iterator.close()
            self._latency_ratio = None

    def relaunch_chunks_iterator(self, *args, **kwargs):
        self.close_chunks_iterator()
        return self.get_chunks_iterator(*args, **kwargs)

    def _get_chunks_iterator(self, chunk_size_sec: float = None):
        audio_chunks_queue = Queue()
        chunk_size_sec = chunk_size_sec or cfg.CHUNK_SIZE_SEC
        args = (
            audio_chunks_queue,
            chunk_size_sec,
            self.pcm_params,
        )
        kwargs = self._get_recorder_proc_kwargs()
        stream_recorder_proc = Process(
            target=self._recorder_proc_func,
            args=args,
            kwargs=kwargs,
        )
        stream_recorder_proc.start()
        logger.info("Start recording")
        try:
            with Ticker(chunk_size_sec) as ticker:
                while True:
                    if not audio_chunks_queue.empty():
                        yield audio_chunks_queue.get()
                        ticker.tick(wait=False)
                        self._latency_ratio = ticker.latency / chunk_size_sec
                    else:
                        time.sleep(cfg.QUEUE_CHECK_DELAY_SEC)
        except (KeyboardInterrupt, GeneratorExit):
            pass
        finally:
            self._chunks_iterator = None
            stream_recorder_proc.terminate()
            stream_recorder_proc.join(5.0)
            stream_recorder_proc.close()
            logger.info("Stop recording")
