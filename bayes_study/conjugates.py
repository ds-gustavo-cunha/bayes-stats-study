#######################
####### IMPORTS #######
from typing import List, Dict, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy.stats import beta, bernoulli
import streamlit as st
from bayes_study.validators.conjugates_validators import (
    BetaBernoulliConjugateParams, 
    BetaBernoulliConjugatePlotDists
)

#####################
####### CLASS #######


class BetaBernoulliConjugate:
    def __init__(self, **params) -> None:
        # validate the input parameters
        validated_params = BetaBernoulliConjugateParams(**params)      
        # define object attributes
        self.prior_alpha = validated_params.prior_alpha
        self.prior_beta = validated_params.prior_beta
        if validated_params.likelihood_dist is not None:
            self.likelihood_prob = np.mean(validated_params.likelihood_dist)
            self.likelihood_trials = len(validated_params.likelihood_dist)
        elif (validated_params.likelihood_prob is not None) and (validated_params.likelihood_trials is not None):
            self.likelihood_prob = validated_params.likelihood_prob
            self.likelihood_trials = validated_params.likelihood_trials
        else:
            raise ValueError(
                "At least `likelihood_dist` param or both "
                "`likelihood_prob and likelihood_trails` params "
                "must not be None!"
                )
        self.sampling_size = validated_params.sampling_size
        # define a dict to control paired samples
        self.sample_iter = dict(prior=0, likelihood=0, posterior=0)
        # sample from prior and likelihood and posterior
        # so as to be able to display report and plot
        # with no updates
        self.sample_prior()
        self.sample_likelihood()
        self.sample_posterior()

    def sample_prior(self):
        # sample from betar prior given inputs
        self.prior_dist = beta(self.prior_alpha, self.prior_beta).rvs(
            size=self.sampling_size
        )
        # add one to sampling count
        self.sample_iter["prior"] += 1

    def sample_likelihood(self):
        # sample from bernoulli likelihood given inputs
        self.likelihood_dist = bernoulli.rvs(
            p=self.likelihood_prob, size=self.likelihood_trials
        )
        self.likelihood_success = np.sum(self.likelihood_dist)
        # add one to sampling count
        self.sample_iter["likelihood"] += 1

    def sample_posterior(self):
        # check if prior and likelihood were already sampled
        # for the given iteration
        if (self.sample_iter["prior"] == self.sample_iter["likelihood"]) and (
            self.sample_iter["prior"] == self.sample_iter["posterior"] + 1
        ):
            # define posterior atributes
            self.posterior_alpha = self.prior_alpha + self.likelihood_success
            self.posterior_beta = self.prior_beta + (
                self.likelihood_trials - self.likelihood_success
            )
            self.posterior_dist = beta(self.posterior_alpha, self.posterior_beta).rvs(
                size=self.sampling_size
            )
            # add one to sampling count
            self.sample_iter["posterior"] += 1
        # prior or likelihood were not sampled
        # for the given iteration
        else:
            # raise
            raise Exception(
                "For a give iteration (update), "
                "prior and likelihood must be sampled before posterior sampling."
            )

    def update_prior(self):
        # update prior distribution params to make them equal to
        # previous iteration posterior
        self.prior_alpha = self.posterior_alpha
        self.prior_beta = self.posterior_beta

    def update_distributions(self):
        # update prior distribution to be equal to
        # previous previous posterior
        self.update_prior()
        # sample prior distribution
        self.sample_prior()
        # sample likelihood distribution
        self.sample_likelihood()
        # sample posterior distribution
        self.sample_posterior()

    def report_dists(self):
        # print report
        print(
            f"Prior: beta({self.prior_alpha}, {self.prior_beta})\n"
            f"Likelihood dist: bernoulli(p={self.likelihood_prob}, size={self.likelihood_trials})\n"
            f"Likelihood iter: success={self.likelihood_success}, failure={self.likelihood_trials-self.likelihood_success}\n"
            f"Posterior: beta({self.posterior_alpha}, {self.posterior_beta})\n"
            f"Sample iteration: {self.sample_iter}"
        )

    def plot_dists(self, **kwargs):
        # validate inputs
        params = BetaBernoulliConjugatePlotDists(**kwargs)
        # define style to use
        plt.style.use("fivethirtyeight")
        # clear axis - sanity check
        params.ax.clear()
        # plot figures
        if params.plot_prior:
            sns.kdeplot(
                x=self.prior_dist,
                label="prior",
                color="b",
                common_norm=True,
                fill=False,
                ax=params.ax,
            )
        sns.kdeplot(
            x=self.likelihood_dist,
            label="likelihood",
            color="y",
            common_norm=True,
            fill=True,
            ax=params.ax,
        )
        if params.plot_posterior:
            sns.kdeplot(
                x=self.posterior_dist,
                label="posterior",
                color="r",
                common_norm=True,
                fill=False,
                ax=params.ax,
            )
        # define plot details
        params.ax.axvline(
            x=self.likelihood_prob,
            ymin=0,
            ymax=1,
            color="y",
            linestyle="--",
            linewidth=1.5,
            label="Likelihood prob",
        )
        prior_msg = f"Prior: beta({self.prior_alpha}, {self.prior_beta})\n"
        posterior_msg = f"Posterior: beta({self.posterior_alpha}, {self.posterior_beta})\n"
        plt.title(
            f"{prior_msg if params.plot_prior else ''}"
            f"Likelihood dist: bernoulli(p={self.likelihood_prob}, size={self.likelihood_trials})\n"
            f"Likelihood iter: success={self.likelihood_success}, failure={self.likelihood_trials-self.likelihood_success}\n"
            f"{posterior_msg if params.plot_posterior else ''}"
            f"Sample iteration: {self.sample_iter['posterior']}",
            loc="left",
        )
        plt.legend(bbox_to_anchor=(1.01, 1))
        plt.xticks([i / 10 for i in range(0, 10 + 1)])
        plt.xlim(-0.01, 1.01)
        # workaround to control streamlit figure size
        # https://stackoverflow.com/questions/71566299/how-to-decrease-plot-size-in-streamlit
        # fig.savefig(
        #     fname="bayes_formulae.png",
        #     transparent=False, dpi='figure', format="png",
        #     pad_inches=0.1, facecolor='auto', edgecolor='auto'
        #     )
        # bayse_formulae_plot = Image.open('figure_name.png')
        # if st_empty_obj is not None:
        #     st_empty_obj.image(
        #         image=bayse_formulae_plot, caption=None,
        #         width=None, use_column_width=False,
        #         clamp=False, channels="RGB", output_format="auto"
        #     )
        if params.st_empty_obj is not None:
            params.st_empty_obj.pyplot(
                fig=params.fig, clear_figure=None, use_container_width=False
            )
        else:
            plt.show()