from envs.atari_wrappers import make_atari


def make_envs(env_params: dict):
    """
    Create training and evaluation environments.

    Args:
        env_params: Dictionary with environment settings. Expected keys:
            - "env_name": str
            - "train_full_episode": bool

    Returns:
        Tuple (train_env, eval_env)
    """
    env_name = env_params["env_name"]
    train_full_episode = env_params["train_full_episode"]

    train_env = make_atari(env_name, terminal_on_life_loss=not train_full_episode)
    eval_env = make_atari(env_name, terminal_on_life_loss=False)

    return train_env, eval_env
