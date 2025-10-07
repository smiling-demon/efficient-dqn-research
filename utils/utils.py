import yaml


def fix_params(env, params):
    agent_params = params["agent_params"]
    agent_params["env_lives"] = env.unwrapped.ale.lives()
    params["agent_params"] = agent_params

    dqn_params = params["dqn_params"]
    dqn_params["use_curl"] = agent_params["use_curl"]
    if agent_params["use_drq"]:
        dqn_params["preprocessor"] = "DrQ"
    if agent_params["use_dueling"]:
        dqn_params["head"] = "Dueling"
    dqn_params["encoder_params"]["in_shape"] = env.observation_space.shape
    dqn_params["head_params"] = {
        "in_dim": dqn_params["encoder_params"]["out_dim"],
        "num_actions": env.action_space.n,
        "atoms": agent_params["atoms"],
        "use_noisy": agent_params["use_noisy"],
        "use_dropout": agent_params["use_dropout"]
    }
    params["dqn_params"] = dqn_params

    loss_params = params["loss_params"]
    loss_params["gamma"] = agent_params["gamma"]
    loss_params["n_step"] = agent_params["n_step"]
    loss_params["atoms"] = agent_params["atoms"]
    loss_params["use_double"] = agent_params["use_double"]

    loss_name = "vanilla"
    if agent_params["use_c51"]:
        if agent_params["use_munchausen"]:
            loss_name = "munchausen_c51"
        else:
            loss_name = "c51"
    elif agent_params["use_qr"]:
        if agent_params["use_munchausen"]:
            loss_name = "munchausen_qr"
        else:
            loss_name = "qr"
    elif agent_params["use_munchausen"]:
        loss_name = "munchausen"
    loss_params["loss_name"] = loss_name
    params["loss_params"] = loss_params

    replay_params = params["replay_params"]
    replay_params["gamma"] = agent_params["gamma"]
    replay_params["n_step"] = agent_params["n_step"]
    params["replay_params"] = replay_params

    return params


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)
