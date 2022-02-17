from datetime import datetime, timedelta
import gym
import random

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.wandb import WandbLoggerCallback


def main():
    cql_config = {
        "env": "HalfCheetahBulletEnv-v0",
        "framework": "tf",
        # Use input produced by expert SAC algo.
        "input": "/Users/jungong/halfcheetah_expert_sac.json",
        "actions_in_input_normalized": True,
        "soft_horizon": False,
        "horizon": 1000,
        "Q_model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [256, 256, 256],
        },
        "policy_model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [256, 256, 256],
        },
        "tau": 0.005,
        "target_entropy": "auto",
        "no_done_at_end": False,
        "n_step": 3,
        "rollout_fragment_length": 1,
        "prioritized_replay": False,
        "train_batch_size": 256,
        "target_network_update_freq": 0,
        "timesteps_per_iteration": 1000,
        "learning_starts": 256,
        "optimization": {
            "actor_learning_rate": 0.0001,
            "critic_learning_rate": 0.0003,
            "entropy_learning_rate": 0.0001,
        },
        "num_workers": 3,
        "num_gpus": 0,
        "metrics_smoothing_episodes": 5,

        # CQL Configs
        "min_q_weight": 5.0,
        "bc_iters": 20000,
        "temperature": 1.0,
        "num_actions": 10,
        "lagrangian": False,

        # Switch on online evaluation.
        "always_attach_evaluation_results": True,
        "evaluation_interval": 5,
        "evaluation_config": {
            "input": "sampler",
        },
        "always_attach_evaluation_results": True,
    }

    ts = datetime.now().strftime('%m%d-%H%M%S')
    results = tune.run(
        "CQL",
        name=f"dataset-{ts}",
        checkpoint_at_end=True,
        checkpoint_freq=5,
        config=cql_config,
        max_failures=5,
        num_samples=1,
        stop={"time_total_s": 3600 * 10},
        verbose=True,
        progress_reporter=CLIReporter(
            metric_columns={
                "training_iteration": "iter",
                "time_total_s": "time_total_s",
                "timesteps_total": "ts",
                "episodes_this_iter": "train_episodes",
                "episode_reward_mean": "reward_mean",
                "evaluation/episode_reward_mean": "eval_reward_mean",
            },
            sort_by_metric=True,
            max_report_frequency=30,
        ),
    )

    results.results_df.to_csv("results.csv")


if __name__ == "__main__":
    main()
