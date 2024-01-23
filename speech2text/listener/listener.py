from __future__ import annotations

import logging
import time
from multiprocessing import Process, Queue

from .pcm_params import WHISPER_PRESET, PcmParams

logger = logging.getLogger(__name__)

QUEUE_CHECK_DELAY_SEC = 0.05
DEFAULT_CHUNK_SIZE_SEC = 0.5


class Listener:
    def __init__(self) -> None:
        self.chunk_size_sec = None
        self.pcm_params: PcmParams = WHISPER_PRESET
        self._recorder_proc = None
        self.set_chunk_size_sec(DEFAULT_CHUNK_SIZE_SEC)

    def set_chunk_size_sec(self, chunk_size_sec):
        self.chunk_size_sec = chunk_size_sec
        return self

    def _get_recorder_proc_args(self):
        raise NotImplementedError

    def gen_chunks(self):
        audio_chunks_queue = Queue()
        args = (audio_chunks_queue, *self._get_recorder_proc_args())
        stream_recorder_proc = Process(
            target=self._recorder_proc_func,
            args=args,
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
