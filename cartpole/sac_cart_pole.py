import ray
import ray
from ray import tune
from ray.tune import CLIReporter


def main():
    config = {
        'env': 'CartPole-v0',
        'framework': 'tf',
        'gamma': 0.95,
        'no_done_at_end': False,
        'target_network_update_freq': 32,
        'tau': 1.0,
        'train_batch_size': 32,
        'optimization': {
            'actor_learning_rate': 0.005,
            'critic_learning_rate': 0.005,
            'entropy_learning_rate': 0.0001
        },

        # Switch on online evaluation.
        "evaluation_interval": 5,
        "evaluation_config": {
            "input": "sampler",
        },
        "always_attach_evaluation_results": True,
    }

    results = tune.run(
        "SAC",
        name=f"sac-cartpole",
        resume=None,
        config=config,
        num_samples=1,
        checkpoint_at_end=True,
        checkpoint_freq=5,
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
        )
    )


if __name__ == "__main__":
    main()
