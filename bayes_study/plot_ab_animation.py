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
    prog=os.path.basename(sys.argv[0]),
    description="Script to display animated AB test.",
    add_help=True,
)

# add arguments
cli_parser.add_argument(
    "--control_prior_alpha",
    type=int,
    default=1,
    nargs="?",
    help="Alpha of prior beta distribution for control group.",
)
cli_parser.add_argument(
    "--control_prior_beta",
    type=int,
    default=1,
    nargs="?",
    help="Beta of prior beta distribution for control group.",
)
cli_parser.add_argument(
    "--control_likelihood_prob",
    type=float,
    default=0.7,
    nargs="?",
    help="Probability of the Bernoulli trials of the likelihood distribution for control group.",
)
cli_parser.add_argument(
    "--control_likelihood_trials",
    type=int,
    default=10,
    nargs="?",
    help="Number of Bernoulli trials of the likelihood distribution for control group.",
)
cli_parser.add_argument(
    "--treatment_prior_alpha",
    type=int,
    default=1,
    nargs="?",
    help="Alpha of prior beta distribution for treatment group.",
)
cli_parser.add_argument(
    "--treatment_prior_beta",
    type=int,
    default=1,
    nargs="?",
    help="Beta of prior beta distribution for treatment group.",
)
cli_parser.add_argument(
    "--treatment_likelihood_prob",
    type=float,
    default=0.75,
    nargs="?",
    help="Probability of the Bernoulli trials of the likelihood distribution for treatment group.",
)
cli_parser.add_argument(
    "--treatment_likelihood_trials",
    type=int,
    default=10,
    nargs="?",
    help="Number of Bernoulli trials of the likelihood distribution for treatment group.",
)
cli_parser.add_argument(
    "--sampling_size",
    type=int,
    default=1_000,
    nargs="?",
    help="Size of the `rvs` sampling that will be used on Beta and Bernoulli distributions.",
)
cli_parser.add_argument(
    "--num_frames",
    type=int,
    default=1_000,
    nargs="?",
    help="Number of frames to display (equal to number of updates)",
)
cli_parser.add_argument(
    "--ms_interval",
    type=int,
    default=1_000,
    nargs="?",
    help="Interval in milli-seconds between plot displays.",
)

# parse CLI arguments
cli_args = cli_parser.parse_args()

# create a Beta-Bernoulli conjugate for control group
control_bbc = BetaBernoulliConjugate(
    prior_alpha=cli_args.control_prior_alpha,
    prior_beta=cli_args.control_prior_beta,
    likelihood_prob=cli_args.control_likelihood_prob,
    likelihood_trials=cli_args.control_likelihood_trials,
    sampling_size=cli_args.sampling_size,
)
# create a Beta-Bernoulli conjugate for treatment group
treatment_bbc = BetaBernoulliConjugate(
    prior_alpha=cli_args.treatment_prior_alpha,
    prior_beta=cli_args.treatment_prior_beta,
    likelihood_prob=cli_args.treatment_likelihood_prob,
    likelihood_trials=cli_args.treatment_likelihood_trials,
    sampling_size=cli_args.sampling_size,
)


def update_plots(frame, control_bbc, treatment_bbc, fig, axs):
    # each time update posterior distributions for both conjugates
    control_bbc.update_distributions()
    treatment_bbc.update_distributions()

    # instanciate a AB testing object
    ab_test = ABTest(control_data=control_bbc, treatment_data=treatment_bbc)
    # calculate AB testing statistics
    ab_test.calculate_ab_stats()
    # plto
    ab_test.plot_ab_dists(axs=axs, fig=fig, stats_title=True)


# if running script directly
if __name__ == "__main__":
    # define plot layout
    fig, axs = plt.subplots(ncols=1, nrows=2, constrained_layout=True, figsize=(10, 12))
    # display animation
    animation = FuncAnimation(
        fig=fig,
        func=partial(
            update_plots,
            control_bbc=control_bbc,
            treatment_bbc=treatment_bbc,
            fig=fig,
            axs=axs,
        ),
        frames=cli_args.num_frames,
        interval=cli_args.ms_interval,  # milliseconds
        repeat=False,
    )
    # display
    plt.show()
