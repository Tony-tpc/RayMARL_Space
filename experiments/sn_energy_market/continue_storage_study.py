"""Validated candidate-scenario change; historical round-72 launcher is archived."""

def change_candidate_seeds(algorithm, seeds):
    """Explicit experiment protocol change, after exact native checkpoint restore."""
    before = algorithm.config.learning_contract()
    assert len(set(seeds)) >= 2 and set(seeds) <= set(before["validation_seeds"])
    config = algorithm.config.copy(copy_frozen=False)
    config.validation_seeds = tuple(seeds)
    config.validate()
    after = config.learning_contract()
    assert {k for k in before if before[k] != after[k]} <= {"validation_seeds"}
    config.freeze()
    algorithm.config = config
    algorithm.env_runner.config = config
    # RLlib Learner keeps its own mutable config for policies_to_train on restore.
    # Preserve that lifecycle and the component's other settings.
    def update_component(component):
        local = component.config.copy(copy_frozen=False)
        local.validation_seeds = tuple(seeds)
        if component.config._is_frozen:
            local.freeze()
        component.config = local

    update_component(algorithm.learner_group)
    for result in algorithm.learner_group.foreach_learner(update_component):
        result.get()
    return after

