import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack


def make_atari(env_name: str, terminal_on_life_loss: bool = True) -> gym.Env:
    """
    Create an Atari environment with standard preprocessing.

    Args:
        env_name: Name of the Atari environment (e.g. "BreakoutNoFrameskip-v4").
        terminal_on_life_loss: Whether to treat life loss as terminal (for training).

    Returns:
        Preprocessed Gymnasium environment with stacked frames.
    """
    env = gym.make(env_name)
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        terminal_on_life_loss=terminal_on_life_loss,
        grayscale_obs=True,
    )
    env = FrameStack(env, num_stack=4)
    return env
