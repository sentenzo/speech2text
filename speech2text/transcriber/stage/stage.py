from ..state import State


class IStage:
    def _check_in_contract(self, state: State, *args, **kwargs):
        state.validate()

    def _check_out_contract(self, state: State, *args, **kwargs):
        state.validate()

    def _apply(self, state: State, *args, **kwargs):
        raise NotImplementedError

    def apply(self, state: State, *args, **kwargs) -> State:
        self._check_in_contract(state, *args, **kwargs)
        self._apply(state, *args, **kwargs)
        self._check_out_contract(state, *args, **kwargs)
