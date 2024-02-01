from dataclasses import dataclass, field

from .stage import (
    AdjustmentStage,
    FinalizationStage,
    IncrementStage,
    IStage,
    RefinementStage,
    SplittingStage,
    TranscriptionStage,
)
from .state import State, Status


@dataclass
class Strategy:
    increment: IStage
    adjust: IStage
    split: IStage
    refine: IStage
    transcribe: IStage
    finalize: IStage

    def process(
        self,
        state: State,
        chunk: bytes | bytearray,
        latency_ratio: float = 0.0,
    ) -> State:
        """
        FINALIZED
        ↓  increment()
        INCREMENTED
        ↓  adjust()
        ADJUSTED
        ↓  split() ------+
        SPLITTED         |
        ↓  refine()      ↓
        REFINED       SKIPPED
        ↓  transcribe()  |
        TRANSCRIBED      |
        ↓  finalize  ←---+
        FINALIZED
        ↓
        ...
        """
        state = self.increment.apply(state, chunk, latency_ratio)
        state = self.adjust.apply(state)
        state = self.split.apply(state)
        if state.status != Status.SKIPPED:
            state = self.refine.apply(state)
            state = self.transcribe.apply(state)
        state = self.finalize.apply(state)
        return state


DEFAULT_STRATEGY = Strategy(
    IncrementStage(),
    AdjustmentStage(),
    SplittingStage(),
    RefinementStage(),
    TranscriptionStage(),
    FinalizationStage(),
)
