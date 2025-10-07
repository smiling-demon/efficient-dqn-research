from agents import Agent
from envs import make_envs
from utils import fix_params, load_config


def main():
    params = load_config("configs/config.yaml")

    # Create environments
    env, eval_env = make_envs(params["env_params"])

    # Update params with env info (e.g., obs/action shapes)
    params = fix_params(env, params)

    # Initialize and train agent
    agent = Agent(env, eval_env, params)
    run_params = params["run_params"]

    agent.train(
        num_frames=run_params["steps"],
        max_train_time=run_params["max_train_time"],
    )


if __name__ == "__main__":
    main()
