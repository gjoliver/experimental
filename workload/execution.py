import logging
from datetime import datetime

import ray
from ray import tune

from config import construct_trial_config
from utils import ImpatientCLIReporter


logger = logging.getLogger(__name__)


def initialize_ray(ignore_reinit_error=True):
    ray.init(
        ignore_reinit_error=ignore_reinit_error,
        local_mode=False,
        runtime_env={
            "working_dir": "."
        },
    )


class Runner:
    def __init__(
        self,
        tune_config_overrides={},
        wandb_project="default",
        wandb_group=None,
        wandb_log_config=False,
        wandb_excludes=[],
    ):
        self.tune_config_overrides = tune_config_overrides
        self.wandb_project = wandb_project
        self.wandb_group = wandb_group
        self.wandb_log_config = wandb_log_config
        self.wandb_excludes = wandb_excludes

    def _construct_default_tune_config(self, trial_min_time_s_per_iteration):
        experiment_name = f"TestExperiment-{datetime.now().strftime('%Y%M%d-%H%M%S')}"
        tune_config = {
            "name": experiment_name,
            "trial_name_creator": lambda trial: (f"{experiment_name}-{str(trial).split('_')[-1]}"),
            "trial_dirname_creator": lambda trial: f"{str(trial).split('_')[-1]}",
            "num_samples": 1,
            "max_concurrent_trials": 5,
            "stop": {"time_total_s": 3600 * 24 * 14},
            "verbose": 1,
            "progress_reporter": ImpatientCLIReporter(
                min_time_s_between_reports=trial_min_time_s_per_iteration - 10,
                max_time_s_between_reports=trial_min_time_s_per_iteration + 60,
                metric_columns={
                    "training_iteration": "iter",
                    "time_total_h": "time_total_h",
                    "timesteps_total": "ts",
                    "info/learner_queue/size_mean": "learner_queue_mean",
                    "info/learner/main_agent/learner_stats/entropy": "entropy",
                    "info/learner/main_agent/learner_stats/vf_explained_var": "vf_explained_var",
                    "episodes_this_iter": "train_episodes",
                    "evaluation/episodes_this_iter": "eval_episodes",
                },
                sort_by_metric=True,
            ),
            "sync_config": None,
            "checkpoint_freq": 5,
            "checkpoint_at_end": True,
            "resume": "AUTO",
            "max_failures": -1,
        }

        return tune_config

    def run(self, algorithm_class, test=False):
        initialize_ray()

        trial_config = construct_trial_config(test)
        tune_config = self._construct_default_tune_config(trial_config["min_time_s_per_iteration"])

        # TODO(jungong) : Figure out wandb integration and cloud checkpointing.
        '''
        tune_config.setdefault("callbacks", []).append(
            wandb_integration.WandbLoggerCallback(
                api_key=FLAGS.wandb_api_key,
                entity=f"aia-{FLAGS.game}",
                project=self.wandb_project,
                group=self.wandb_group,
                log_config=self.wandb_log_config,
                excludes=self.wandb_excludes,
                resume=True,
            )
        )

        tune_config["sync_config"] = tune.SyncConfig(
            syncer=checkpointing.CustomCommandSyncer(
                sync_up_template="gsutil -mq rsync -r {source} {target}",
                sync_down_template="mkdir -p {target}; gsutil -mq rsync -r {source} {target}",
                delete_template="gsutil rm -r {target}",
            ),
            upload_dir=f"gs://riot-ai-central/{FLAGS.game}",
            sync_on_checkpoint=False,
        )
        '''

        logger.info(trial_config)
        logger.info(tune_config)

        tune.run(run_or_experiment=algorithm_class, config=trial_config, **tune_config)
