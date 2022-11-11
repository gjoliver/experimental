# Standard Library
import subprocess

# Third Party
from ray.tune.syncer import Syncer


class CustomCommandSyncer(Syncer):
    def __init__(
        self,
        sync_up_template: str,
        sync_down_template: str,
        delete_template: str,
        sync_period: float = 300.0,
    ):
        self.sync_up_template = sync_up_template
        self.sync_down_template = sync_down_template
        self.delete_template = delete_template

        super().__init__(sync_period=sync_period)

    def sync_up(self, local_dir: str, remote_dir: str, exclude: list = None) -> bool:
        cmd_str = self.sync_up_template.format(
            source=local_dir,
            target=remote_dir,
        )
        subprocess.check_call(cmd_str, shell=True)

        return True

    def sync_down(self, remote_dir: str, local_dir: str, exclude: list = None) -> bool:
        cmd_str = self.sync_down_template.format(
            source=remote_dir,
            target=local_dir,
        )
        try:
            subprocess.check_call(cmd_str, shell=True)
        except subprocess.CalledProcessError as e:
            print(f"Syncer failed to sync_down, proceeding in spite of this. Error:\n\t{e}")

        return True

    def delete(self, remote_dir: str) -> bool:
        cmd_str = self.delete_template.format(
            target=remote_dir,
        )
        subprocess.check_call(cmd_str, shell=True)

        return True

    def retry(self):
        raise NotImplementedError

    def wait(self):
        pass
