# import required libs
import os
import sys
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial
from bayes_study.conjugates import BetaBernoulliConjugate
from bayes_study.ab_test import ABTest


# create a ArgumentParser
cli_parser = argparse.ArgumentParser(
    description="Script to display animated distribution update."
)

# add arguments
cli_parser.add_argument("--prior_alpha", type=int, default=1, nargs="?")
cli_parser.add_argument("--prior_beta", type=int, default=1, nargs="?")
cli_parser.add_argument("--likelihood_prob", type=float, default=0.7, nargs="?")
cli_parser.add_argument("--likelihood_trials", type=int, default=10, nargs="?")
cli_parser.add_argument("--sampling_size", type=int, default=1_000, nargs="?")
cli_parser.add_argument("--plot_prior", type=bool, default=True, nargs="?")
cli_parser.add_argument("--plot_posterior", type=bool, default=True, nargs="?")
cli_parser.add_argument("--num_frames", type=int, default=1_000, nargs="?")
cli_parser.add_argument("--ms_interval", type=int, default=1_000, nargs="?")

# parse CLI arguments
cli_args = cli_parser.parse_args()

# create a Beta-Bernoulli conjugate distributions object
# the same one from previous cel
bbc = BetaBernoulliConjugate(
    prior_alpha=cli_args.prior_alpha,
    prior_beta=cli_args.prior_beta,
    likelihood_prob=cli_args.likelihood_prob,
    likelihood_trials=cli_args.likelihood_trials,
    sampling_size=cli_args.sampling_size,
)


def update_plots(frame, bbc, fig, ax, cli_args):
    # update posterior from prior and likelihood
    bbc.update_distributions()

    # plot
    bbc.plot_dists(
        ax=ax,
        fig=fig,
        plot_prior=cli_args.plot_prior,
        plot_posterior=cli_args.plot_posterior,
    )


# if running script directly
if __name__ == "__main__":
    # define plot layout
    fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 7))
    # display animation
    animation = FuncAnimation(
        fig=fig,
        func=partial(update_plots, bbc=bbc, fig=fig, ax=ax, cli_args=cli_args),
        frames=cli_args.num_frames,
        interval=cli_args.ms_interval,  # milliseconds
        repeat=False,
    )
    # display
    plt.show()
