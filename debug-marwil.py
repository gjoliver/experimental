from datetime import datetime, timedelta
import gym
import random

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.wandb import WandbLoggerCallback
import wandb


def main():
    # Defaults
    config = {
        "env": "HalfCheetahBulletEnv-v0",
        "framework": "tf",
        # Use input produced by expert SAC algo.
        "input": ["~/halfcheetah_expert_sac.zip"],
        "input_evaluation": [],
        "actions_in_input_normalized": True,
        "always_attach_evaluation_results": True,
        "num_gpus": 0,
        "model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [256, 256, 256],
        },
        "logstd_coeff": tune.grid_search([0.0, 0.8, 0.9]),
        "evaluation_num_workers": 1,
        "evaluation_interval": 5,
        "evaluation_config": {
            "input": "sampler",
        }
    }

    ts = datetime.now().strftime('%m%d-%H%M%S')
    results = tune.run(
        "MARWIL",
        name=f"debug-bc-{ts}",
        resume=None,
        config=config,
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
        callbacks=[
            WandbLoggerCallback(
                name=f"run-{ts}",
                project="debug-marwil",
                api_key="ec234a418d6c19a4de1c3906f3561e7c1214d933",
                log_config=False,
                settings=wandb.Settings(start_method="fork")),
        ],
    )

    results.results_df.to_csv("results.csv")


if __name__ == "__main__":
    main()
