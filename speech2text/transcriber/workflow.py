from typing import List

from speech2text.audio_data import WHISPER_PCM_PARAMS, PcmParams

from .state import State
from .strategy import DEFAULT_STRATEGY, IStrategy


class Workflow:
    def __init__(
        self,
        *,
        strategy: IStrategy = DEFAULT_STRATEGY,
        input_pcm_params: PcmParams = WHISPER_PCM_PARAMS,
    ) -> None:
        self.strategy = strategy
        self.strategy.cold_start()
        self.state = State(input_pcm_params)

    def process_chunk(
        self,
        chunk: bytes | bytearray,
        latency_ratio: float = 0.0,
    ):
        self.state = self.strategy.process_chunk(
            self.state, chunk, latency_ratio
        )

    def get_finalized_text(self, flush_blocks=False) -> List[str]:
        lines = [block.text for block in self.state.finalized]
        if flush_blocks and self.state.finalized:
            self.state.finalized = []
        return lines

    def get_ongoing_text(self) -> str:
        return self.state.ongoing.text or ""
