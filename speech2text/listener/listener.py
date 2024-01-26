from __future__ import annotations

import logging
import time
from multiprocessing import Process, Queue

import speech2text.config as cfg
from speech2text.pcm_params import WHISPER_PRESET, PcmParams

logger = logging.getLogger(__name__)


class Listener:
    def __init__(self, pcm_params: PcmParams = WHISPER_PRESET) -> None:
        self.pcm_params: PcmParams = pcm_params
        self._recorder_proc = None

    def _get_recorder_proc_kwargs(self):
        raise NotImplementedError

    def get_chunks_iterator(self, chunk_size_sec: float = None):
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
            while True:
                time.sleep(cfg.QUEUE_CHECK_DELAY_SEC)
                if not audio_chunks_queue.empty():
                    yield audio_chunks_queue.get()
        except KeyboardInterrupt:
            pass
        finally:
            logger.info("Stop recording")
