import abc


class Agent(abc.ABC):
    # abstract base class defining the core interface for all reinforcement learning agents.

    def reset(self):
        # reset any internal states at the start of an episode (for stateful agents).
        pass

    @abc.abstractmethod
    def train(self, training: bool = True):
        """switch the agent between training and evaluation modes."""

    @abc.abstractmethod
    def update(self, replay_buffer, logger, step: int):
        """perform one learning update using experience from the replay buffer."""

    @abc.abstractmethod
    def act(self, obs, sample: bool = False):
        """select an action given the current observation."""
