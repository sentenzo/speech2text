from ..state import State


class IStrategy:
    def cold_start(self, state: State):
        raise NotImplementedError

    def process_chunk(
        self,
        state: State,
        chunk: bytes | bytearray,
        latency_ratio: float = 0.0,
    ) -> State:
        raise NotImplementedError
