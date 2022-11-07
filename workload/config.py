import gym
from ray.rllib.algorithms import callbacks
from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.algorithms.appo.appo_tf_policy import APPOTF2Policy
from ray.rllib.policy import policy
from ray.tune import registry

from utils import Matchmaker, MetricsCallbacks


# Fake 2 player game.
ENV_NUM_PLAYERS = 2
ENV_NAME = "MockMultiAgentGameEnv"
MultiAgentRandomEnv = make_multi_agent(
    lambda _: RandomEnv({
        "observation_space": gym.spaces.Box(low=-1, high=1, shape=(1000,)),
        "action_space": gym.spaces.Discrete(n=25),
        "p_done": 0.01
    })
)


# TODO(jungong) : Add self-play.
def construct_trial_config(test: bool):
    registry.register_env(ENV_NAME, lambda cfg: MultiAgentRandomEnv(cfg))

    if test:
        num_gpus = 0
        num_workers = 2
        evaluation_num_workers = 1
    else:
        num_gpus = 1
        num_workers = 80
        evaluation_num_workers = 16

    return {
        # Basics
        "env": ENV_NAME,
        "env_config": {
            "num_agents": ENV_NUM_PLAYERS,
        },
        "disable_env_checking": True,
        "framework": "tf2",
        "_disable_execution_plan_api": True,
        "eager_tracing": True,
        "num_gpus": num_gpus,
        "num_workers": num_workers,
        "evaluation_num_workers": evaluation_num_workers,
        "num_envs_per_worker": 1,
        "recreate_failed_workers": True,
        "restart_failed_sub_environments": True,
        "num_consecutive_worker_failures_tolerance": 100,
        "min_time_s_per_iteration": 60 * 6,
        "enable_connectors": True,
        # Multiagent (self-play)
        "multiagent": {
            "policies": {
                "main_agent": policy.PolicySpec(policy_class=APPOTF2Policy),
                "random": policy.PolicySpec(policy_class=APPOTF2Policy),
            },
            "policy_mapping_fn": Matchmaker("main_agent").policy_mapping_fn,
            "policies_to_train": ["main_agent"],
            "count_steps_by": "agent_steps",
        },
        # Learning
        "batch_mode": "complete_episodes",
        "vtrace_drop_last_ts": False,
        "rollout_fragment_length": 16,
        "train_batch_size": 1024,
        "gamma": 0.9975,
        "vf_loss_coeff": 0.5,
        "entropy_coeff": 3e-3,
        "clip_param": 0.3,
        "grad_clip": 1,
        # "lr": 1e-5,
        "lr_schedule": [[0, 1e-4], [1e9, 1e-5]],
        "_tf_policy_handles_more_than_one_loss": True,
        "_separate_vf_optimizer": True,
        "_lr_vf": 2e-4,
        # Evaluation
        "enable_async_evaluation": True,
        "evaluation_duration": "auto",
        "evaluation_parallel_to_training": True,
        "evaluation_interval": 1,
        "evaluation_config": {
            "env_config": {"evaluation": True},
            "explore": False,  # True => RLlib policy samples from action probs, False => argmax
            "multiagent": {
                "policy_mapping_fn": Matchmaker("random").policy_mapping_fn,
            },
        },
        # Callbacks
        "callbacks": callbacks.MultiCallbacks(
            [
                MetricsCallbacks,
            ]
        ),
    }
