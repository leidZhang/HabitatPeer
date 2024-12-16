from dataclasses import dataclass

import magnum as mn
import numpy as np

import habitat
from habitat.config.default_structured_configs import ActionConfig, DiscreteNavigationActionConfig
from habitat.tasks.nav.nav import SimulatorTaskAction


# This is the configuration for our action.
@dataclass
class TurnDoubleActionConfig(ActionConfig):
    turn_angle_deg: float = 0.0
    noise_amount: float = 0.0


def _turn_right_body(
    sim,
    turn_angle_deg: float,
    noise_amount: float,
) -> None:
    # Get agent state
    agent_state = sim.get_agent_state()
    normalized_quaternion = agent_state.rotation
    agent_mn_quat = mn.Quaternion(
        normalized_quaternion.imag, normalized_quaternion.real
    )

    turn_angle = np.random.uniform(
        (1 - noise_amount) * turn_angle_deg,
        (1 + noise_amount) * turn_angle_deg,
    )
    turn_angle = mn.Deg(turn_angle)
    rotation = mn.Quaternion.rotation(turn_angle, mn.Vector3.y_axis())
    new_rotation = rotation * agent_mn_quat
    sim.set_agent_state(
        agent_state.position,
        [*new_rotation.vector, new_rotation.scalar],
        reset_sensors=False,
    )


@habitat.registry.register_task_action
class TurnRightDouble(SimulatorTaskAction):
    def __init__(self, *args, config, sim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim = sim
        self._turn_angle_deg = config.turn_angle_deg
        self._noise_amount = config.noise_amount

    def _get_uuid(self, *args, **kwargs) -> str:
        return "turn_right_double"

    def step(self, *args, **kwargs):
        _turn_right_body(self._sim, self._turn_angle_deg * 2, self._noise_amount)