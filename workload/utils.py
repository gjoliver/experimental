import time

from ray.rllib.algorithms import callbacks
from ray.tune import progress_reporter


class MetricsCallbacks(callbacks.DefaultCallbacks):
    def on_train_result(self, algorithm, result, **kwargs):
        result["time_total_h"] = result["time_total_s"] / 3600


class Matchmaker:
    def __init__(self, policy_name):
        self.policy_name = policy_name

    def policy_mapping_fn(self, agent_id, episode, worker, **kwargs):
        return self.policy_name


class ImpatientCLIReporter(progress_reporter.CLIReporter):
    """Extends CLIReporter to improve quality-of-life for our workloads.

    Reports immediately whenever a new `training_iteration` is observed, subject to some min and max
    time between reports (the former to allow us to prevent spam when there are multiple trials, the
    latter so that we can force reporting to occur despite stalled trials).
    """

    def __init__(self, min_time_s_between_reports, max_time_s_between_reports, **kwargs):
        super().__init__(**kwargs)
        self._min_time_s_between_reports = min_time_s_between_reports
        self._max_time_s_between_reports = max_time_s_between_reports
        self._last_iterations_reported = {}
        self._last_report_due_to_time_elapsed = True

    def should_report(self, trials, done):
        time_since_last_report = time.time() - self._last_report_time
        iterations = {
            str(trial): trial._last_result.get("training_iteration", None) for trial in trials
        }
        if (
            time_since_last_report >= self._min_time_s_between_reports
            or self._last_report_due_to_time_elapsed
        ) and iterations != self._last_iterations_reported:
            self._last_iterations_reported = iterations
            self._last_report_due_to_time_elapsed = False
            self._last_report_time = time.time()
            return True
        elif time_since_last_report > self._max_time_s_between_reports:
            self._last_report_due_to_time_elapsed = True
            self._last_report_time = time.time()
            return True

        return done
