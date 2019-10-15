import os
import logging
import click
import tensorflow as tf
from ..runners import abstraction_learn_actions_tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel(logging.ERROR)


@click.command()
@click.option("--dimensionality", default=2, type=click.INT)
@click.option("--num-hidden-states", default=10, type=click.INT)
@click.option("--learning-rate", default=0.01, type=click.FLOAT)
@click.option("--num-steps", default=10000, type=click.INT)
@click.option("--validation-freq", default=200, type=click.INT)
@click.option("--minibatches", default=False, is_flag=True)
@click.option("--batch-size", default=100, type=click.INT)
@click.option("--mu-init-sd", default=1.0, type=click.FLOAT)
@click.option("--cov-init-sd", default=1.0, type=click.FLOAT)
@click.option("--show-graphs", default=False, is_flag=True)
@click.option("--gpu", default=None, type=click.STRING)
def main(*args, **kwargs):
    return abstraction_learn_actions_tf.main(*args, **kwargs)


if __name__ == "__main__":
    main()
