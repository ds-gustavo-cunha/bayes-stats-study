#######################
####### IMPORTS #######
from typing import List, Dict, Union, Any
from pydantic import BaseModel, Field, field_validator, PrivateAttr
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta, bernoulli
from bayes_study.validators.conjugates_validators import (
    BetaBernoulliConjugatePlotDists,
    _validate_likelihood_dist,
)


#####################
####### CLASS #######
class BetaBernoulliConjugate(BaseModel, extra="forbid"):
    # public attributes
    prior_alpha: int = Field(
        default=..., ge=1, description="Alpha of prior beta distribution."
    )
    prior_beta: int = Field(
        default=..., ge=1, description="Beta of prior beta distribution."
    )
    likelihood_dist: Union[List[int], None] = Field(
        default=None,
        description=(
            "List with Bernoulli trials that will be used to "
            "estimate Bernoulli likelihood and take adventage "
            "of conjugate distributions."
        ),
    )
    likelihood_prob: Union[float, None] = Field(
        default=None,
        ge=0.0,
        lt=1.0,
        description=(
            "Probability of the Bernoulli trials of the likelihood distribution."
        ),
    )
    likelihood_trials: Union[int, None] = Field(
        default=None,
        ge=1,
        description=("Number of Bernoulli trials of the likelihood distribution."),
    )
    sampling_size: int = Field(
        default=10_000,
        ge=10,
        description=(
            "Size of the `rvs` sampling that will be used on "
            "Beta and Bernoulli distributions."
        ),
    )
    # private attributes
    _posterior_alpha: Union[int, None] = PrivateAttr(default=None)
    _posterior_beta: Union[int, None] = PrivateAttr(default=None)
    _likelihood_success: Union[int, None] = PrivateAttr(default=None)
    _sample_iter: Union[Dict, None] = PrivateAttr(default=None)
    _prior_dist: Union[List[float], None] = PrivateAttr(default=None)
    _likelihood_dist: Union[List[float], None] = PrivateAttr(default=None)
    _posterior_dist: Union[List[float], None] = PrivateAttr(default=None)

    @field_validator(__field="likelihood_dist", mode="before")
    @classmethod
    def validate_likelihood_dist(cls, value: Union[List[int], None]):
        return _validate_likelihood_dist(value=value)

    def __init__(
        self,
        prior_alpha: int,
        prior_beta: int,
        likelihood_dist: Union[List[int], None] = None,
        likelihood_prob: Union[float, None] = None,
        likelihood_trials: Union[int, None] = None,
        sampling_size: int = 10_000,
    ) -> None:
        """
        Beta-Bernoulli conjugate distributions object to
        perform bayes update and easily get into posterior
        distribution (another beta distribution).

        Args
            prior_alpha: int
                Alpha of prior beta distribution.
            prior_beta: int
                Beta of prior beta distribution.
            likelihood_dist: Union[List[int], None] = None
                List with Bernoulli trials that will be used to
                estimate Bernoulli likelihood and take adventage
                of conjugate distributions.
            likelihood_prob: Union[float, None] = None
                Probability of the Bernoulli trials of the
                likelihood distribution.
            likelihood_trials: Union[int, None] = None,
                Number of Bernoulli trials of the
                likelihood distribution.
            sampling_size: int = 10_000
                Size of the `rvs` sampling that will be used on
                Beta and Bernoulli distributions.
        """
        # pydantic class
        super().__init__(
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            likelihood_dist=likelihood_dist,
            likelihood_prob=likelihood_prob,
            likelihood_trials=likelihood_trials,
            sampling_size=sampling_size,
        )
        # if a likelihood distribution was input
        if self.likelihood_dist is not None:
            self.likelihood_prob = np.mean(np.array(self.likelihood_dist))
            self.likelihood_trials = len(np.array(self.likelihood_dist))
        # a likelihood distribution was not input
        # and likelihood probability and trails are both null
        elif (self.likelihood_prob is None) or (self.likelihood_trials is None):
            raise ValueError(
                "At least `likelihood_dist` param or both "
                "`likelihood_prob and likelihood_trails` params "
                "must not be None!"
            )
        # define a dict to control paired samples
        self._sample_iter = dict(prior=0, likelihood=0, posterior=0)
        # sample from prior and likelihood and posterior
        # so as to be able to display report and plot
        # with no updates
        self.sample_prior()
        self.sample_likelihood()
        self.sample_posterior()

    def sample_prior(self):
        """
        Sample from prior Beta distribution.
        """
        # sample from betar prior given inputs
        self._prior_dist = beta(self.prior_alpha, self.prior_beta).rvs(
            size=self.sampling_size
        )
        # add one to sampling count
        self._sample_iter["prior"] += 1

    def sample_likelihood(self):
        """
        Sample from Bernoulli likelihood distribution.
        """
        # sample from bernoulli likelihood given inputs
        self._likelihood_dist = bernoulli.rvs(
            p=self.likelihood_prob, size=self.likelihood_trials
        )
        self._likelihood_success = np.sum(self._likelihood_dist)
        # add one to sampling count
        self._sample_iter["likelihood"] += 1

    def sample_posterior(self):
        """
        Sample from posterior given the Beta prior and
        Bernoulli likelihood conjugates.
        """
        # check if prior and likelihood were already sampled
        # for the given iteration
        if (self._sample_iter["prior"] == self._sample_iter["likelihood"]) and (
            self._sample_iter["prior"] == self._sample_iter["posterior"] + 1
        ):
            # define posterior atributes
            self._posterior_alpha = self.prior_alpha + self._likelihood_success
            self._posterior_beta = self.prior_beta + (
                self.likelihood_trials - self._likelihood_success
            )
            self._posterior_dist = beta(
                self._posterior_alpha, self._posterior_beta
            ).rvs(size=self.sampling_size)
            # add one to sampling count
            self._sample_iter["posterior"] += 1
        # prior or likelihood were not sampled
        # for the given iteration
        else:
            # raise
            raise Exception(
                "For a give iteration (update), "
                "prior and likelihood must be sampled before posterior sampling."
            )

    def update_prior(self):
        """
        Update prior so as to be equal to previous posterior.
        """
        # update prior distribution params to make them equal to
        # previous iteration posterior
        self.prior_alpha = self._posterior_alpha
        self.prior_beta = self._posterior_beta

    def update_distributions(self):
        """
        Call the following methods in sequence:
            (1) self.update_prior()
            (2) self.sample_prior()
            (3) self.sample_likelihood()
            (4) self.sample_posterior()
        """
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
            f"Likelihood iter: success={self._likelihood_success}, failure={self.likelihood_trials-self._likelihood_success}\n"
            f"Posterior: beta({self._posterior_alpha}, {self._posterior_beta})\n"
            f"Sample iteration: {self._sample_iter}"
        )

    def plot_dists(
        self,
        fig,
        ax,
        plot_prior: bool = True,
        plot_posterior: bool = True,
    ):
        """
        Plot prior, likelihood and/or posterior distributions.

        Args:
            fig: matplotlib.figure.Figure
                plt figure to plot
            ax: matplotlib.axes._axes.Axes
                plt ax to plot
            plot_prior: bool = True
                Boolean to indicate whether to plot prior distribution or not.
            plot_posterior: bool = True
                Boolean to indicate whether to plot posterior distribution or not.
        """
        # validate inputs
        params = BetaBernoulliConjugatePlotDists(
            fig=fig,
            ax=ax,
            plot_prior=plot_prior,
            plot_posterior=plot_posterior,
        )
        # define style to use
        plt.style.use("fivethirtyeight")
        # define font
        plt.rcParams["font.family"] = "monospace"
        # clear axis - sanity check
        params.ax.clear()
        # plot figures
        if params.plot_prior:
            sns.kdeplot(
                x=self._prior_dist,
                label="prior",
                color="b",
                common_norm=True,
                fill=False,
                ax=params.ax,
                warn_singular=False,  # zero-variance warning, kde not plot
            )
        sns.kdeplot(
            x=self._likelihood_dist,
            label="likelihood",
            color="y",
            common_norm=True,
            fill=True,
            ax=params.ax,
            warn_singular=False,  # zero-variance warning, kde not plot
        )
        if params.plot_posterior:
            sns.kdeplot(
                x=self._posterior_dist,
                label="posterior",
                color="r",
                common_norm=True,
                fill=False,
                ax=params.ax,
                warn_singular=False,  # zero-variance warning, kde not plot
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
        prior_msg = f"Prior:              beta({self.prior_alpha},{self.prior_beta})\n"
        posterior_msg = f"Posterior:          beta({self._posterior_alpha},{self._posterior_beta})\n"
        plt.title(
            f"{prior_msg if params.plot_prior else ''}"
            f"Likelihood dist:    bernoulli(p={self.likelihood_prob},size={self.likelihood_trials})\n"
            f"Likelihood iter:    success={self._likelihood_success},failure={self.likelihood_trials-self._likelihood_success}\n"
            f"{posterior_msg if params.plot_posterior else ''}"
            f"Sample iteration:   {self._sample_iter['posterior']}",
            loc="left",
        )
        plt.ylabel("Probability\ndensity", loc="bottom")
        plt.xlabel("Parameter Value", loc="left")
        plt.legend(bbox_to_anchor=(1.01, 1))
        plt.xticks([i / 10 for i in range(0, 10 + 1)])
        plt.xlim(-0.01, 1.01)
