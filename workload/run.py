import argparse
import logging
import sys

from ray.rllib.algorithms.appo.appo import APPO

from execution import Runner


parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true")


def main():
    FLAGS = parser.parse_args()

    if FLAGS.test:
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        root.addHandler(handler)

    runner = Runner()
    runner.run(algorithm_class=APPO, test=FLAGS.test)


if __name__ == "__main__":
    main()
