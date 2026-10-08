"""Native module collection filters training-only modules from weight broadcasts."""

from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule


class TrainingMultiRLModule(MultiRLModule):
    def setup(self):
        super().setup()
        # RLlib 2.59's RLModuleSpec.build does not pass learner_only to the constructor.
        for mid, module in self._rl_modules.items():
            module.learner_only = self.rl_module_specs[mid].learner_only

    def get_state(self, *args, inference_only=False, **kwargs):
        state = super().get_state(*args, inference_only=inference_only, **kwargs)
        if inference_only:
            state = {mid: value for mid, value in state.items() if not self[mid].learner_only}
        return state
