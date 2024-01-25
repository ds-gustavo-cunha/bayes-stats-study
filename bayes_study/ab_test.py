#######################
####### IMPORTS #######
from typing import List, Dict, Union
from pydantic import BaseModel, Field, field_validator, PrivateAttr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta, bernoulli
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import zt_ind_solve_power
from bayes_study.conjugates import BetaBernoulliConjugate
from bayes_study.validators.ab_test_validators import (
    CalculateBayesianStatsParams,
    CalculateFrequentistStatsParams,
    PlotAbDistsParams,
)


#####################
####### CLASS #######
class ABTest(BaseModel, extra="forbid"):
    # public attributes
    control_data: BetaBernoulliConjugate = Field(
        default=...,
        description=(
            "A bayes_study.conjugates.BetaBernoulliConjugate object "
            "with control data."
        ),
    )
    treatment_data: BetaBernoulliConjugate = Field(
        default=...,
        description=(
            "A bayes_study.conjugates.BetaBernoulliConjugate object "
            "with treatment data."
        ),
    )
    # private attributes
    _bayes_stats: Union[Dict, None] = PrivateAttr(default=dict())
    _freq_stats: Union[Dict, None] = PrivateAttr(default=dict())
    _ab_stats: Union[Dict, None] = PrivateAttr(default=dict())

    def __init__(
        self,
        control_data: BetaBernoulliConjugate,
        treatment_data: BetaBernoulliConjugate,
    ):
        """
        Class to create AB test comparision between frequentist and bayesian approach.

        Args
            control_data: BetaBernoulliConjugate
                A bayes_study.conjugates.BetaBernoulliConjugate object instance
                with control data.
            treatment_data: BetaBernoulliConjugate
                A bayes_study.conjugates.BetaBernoulliConjugate object instance
                with treatment data.
        """
        # pydantic class
        super().__init__(control_data=control_data, treatment_data=treatment_data)

    def calculate_bayesian_stats(
        self, credible_interval: int = 90, rope: int = 3
    ) -> Dict:
        """
        Calculate bayesian statistics of AB test.

        Args
            credible_interval: int = 90
                Bayesian credible interval.
            rope: int = 3
                Bayesian Region Of Practical Equivalence (ROPE).
        """
        # validate inputs
        validated_params = CalculateBayesianStatsParams(
            credible_interval=credible_interval, rope=rope
        )
        # calculate probab of treatment being better than control:
        # average proportion of random samples where
        # treatment posteriori is better than control posteriori
        self._bayes_stats["probab_t_better_c"] = np.mean(
            self.treatment_data._posterior_dist > self.control_data._posterior_dist
        )
        # calculate expected loss:
        # expected loss if choose treatment and treatment is worse
        self._bayes_stats["expected_loss"] = np.mean(
            (self.control_data._posterior_dist - self.treatment_data._posterior_dist)
            / self.treatment_data._posterior_dist
        )
        # Calculate uplift:
        # expected improvement if choose treatment and treatment is better
        self._bayes_stats["uplift_t"] = np.mean(
            (self.treatment_data._posterior_dist - self.control_data._posterior_dist)
            / self.control_data._posterior_dist
        )

        # sample lift from distribution
        self._bayes_stats["lift_dist"] = (
            self.treatment_data._posterior_dist / self.control_data._posterior_dist - 1
        )
        # define credible interval boundaries
        self._bayes_stats["lift_cred_inter"] = validated_params.credible_interval
        tail_limits = (100 - self._bayes_stats["lift_cred_inter"]) / 2
        self._bayes_stats["lift_cred_lower_q"] = tail_limits / 100
        self._bayes_stats["lift_cred_upper_q"] = (
            1 - self._bayes_stats["lift_cred_lower_q"]
        )
        # calculate lift difference and credible interval
        self._bayes_stats["lift_mean"] = np.mean(self._bayes_stats["lift_dist"])
        # calculate lift credible interval
        (
            self._bayes_stats["lift_cred_lower_limit"],
            self._bayes_stats["lift_cred_upper_limit"],
        ) = np.quantile(
            a=self._bayes_stats["lift_dist"],
            q=[
                self._bayes_stats["lift_cred_lower_q"],
                self._bayes_stats["lift_cred_upper_q"],
            ],
        )
        # define lower and upper ROPE limits
        self._bayes_stats["rope"] = validated_params.rope / 100
        (
            self._bayes_stats["lift_rope_lower_limit"],
            self._bayes_stats["lift_rope_upper_limit"],
        ) = (-self._bayes_stats["rope"], self._bayes_stats["rope"])

        return self._bayes_stats

    def calculate_frequentist_stats(self, significance_level: int = 5) -> Dict:
        """
        Calculate frequentist statistics of AB test.

        Args
            significance_level: int = 5
                Frequentist significance level for a hypothesis test.
        """
        # validate inputs
        validated_params = CalculateFrequentistStatsParams(
            significance_level=significance_level
        )
        # define required variables
        self._freq_stats["control_obs"] = (
            self.control_data._posterior_alpha + self.control_data._posterior_beta
        )
        self._freq_stats["treatment_obs"] = (
            self.treatment_data._posterior_alpha + self.treatment_data._posterior_beta
        )
        self._freq_stats["signif_level"] = validated_params.significance_level
        # Perform z-test for proportions
        self._freq_stats["z_stat"], self._freq_stats["p_value"] = proportions_ztest(
            count=[
                self.control_data._posterior_alpha,
                self.treatment_data._posterior_alpha,
            ],
            nobs=[self._freq_stats["control_obs"], self._freq_stats["treatment_obs"]],
            alternative="two-sided",
        )
        # Calculate the observed proportion for each group
        observed_proportion_c = (
            self.control_data._posterior_alpha / self._freq_stats["control_obs"]
        )
        observed_proportion_t = (
            self.treatment_data._posterior_alpha / self._freq_stats["treatment_obs"]
        )
        # Calculate the effect size (difference in proportions)
        self._freq_stats["effect_size"] = observed_proportion_t - observed_proportion_c
        # Calculate the power
        self._freq_stats["power"] = zt_ind_solve_power(
            effect_size=self._freq_stats["effect_size"],
            nobs1=self._freq_stats["control_obs"],
            alpha=self._freq_stats["signif_level"] / 100,
            power=None,  # what needs to be calculated
            ratio=1,
            alternative="two-sided",
        )

        return self._freq_stats

    def calculate_ab_stats(
        self, credible_interval: int = 90, rope: int = 3, significance_level: int = 5
    ) -> Dict:
        """
        Call the following methods in sequence:
            (1) self.calculate_bayesian_stats
            (2) self.calculate_frequentist_stats

        Args
            credible_interval: int = 90
                Bayesian credible interval.
            rope: int = 3
                Bayesian Region Of Practical Equivalence (ROPE).
            significance_level: int = 5
                Frequentist significance level for a hypothesis test.
        """
        # validate inputs
        bayes_params = CalculateBayesianStatsParams(
            credible_interval=credible_interval, rope=rope
        )
        freq_params = CalculateFrequentistStatsParams(
            significance_level=significance_level
        )
        # calculate bayes and frequentist stats
        self.calculate_bayesian_stats(
            credible_interval=bayes_params.credible_interval, rope=bayes_params.rope
        )
        self.calculate_frequentist_stats(
            significance_level=freq_params.significance_level
        )
        self._ab_stats = {**self._bayes_stats, **self._freq_stats}
        # return ab stats except the distributions
        return {k: v for k, v in self._ab_stats.items() if not k.endswith("dist")}

    def plot_ab_dists(self, fig, axs, stats_title: bool = False):
        """
        Plot prior, likelihood and/or posterior distributions.

        Args:
            fig: matplotlib.figure.Figure
                plt figure to plot
            ax: matplotlib.axes._axes.Axes
                plt ax to plot
            stats_title: bool = False
                Boolean to indicate whether to print statistical results
                on plot title.
        """
        # validate inputs
        validated_params = PlotAbDistsParams(fig=fig, axs=axs, stats_title=stats_title)

        # define style to use
        plt.style.use("fivethirtyeight")
        # define font
        plt.rcParams["font.family"] = "monospace"
        # clear axis - sanity check
        validated_params.axs[0].clear()
        validated_params.axs[1].clear()
        # plot treatment and control distributions
        sns.kdeplot(
            x=self.control_data._posterior_dist,
            color="r",
            label="control",
            ax=validated_params.axs[0],
            warn_singular=False,  # zero-variance warning, kde not plot
        )
        sns.kdeplot(
            x=self.treatment_data._posterior_dist,
            color="b",
            label="treatment",
            ax=validated_params.axs[0],
            warn_singular=False,  # zero-variance warning, kde not plot
        )
        # define title variables
        control_report = f"  Control                     Beta({self.control_data._posterior_alpha}, {self.control_data._posterior_beta})"
        treatment_report = f"  Treatment                   Beta({self.treatment_data._posterior_alpha}, {self.treatment_data._posterior_beta})"
        stats_report_title = (
            f"Bayes results:\n"
            f"  Probability of T better C   {self._bayes_stats['probab_t_better_c']*100:.2f}%\n"
            f"  Expected loss               {self._bayes_stats['expected_loss']*100:.2f}%\n"
            f"  Expected uplift             {self._bayes_stats['uplift_t']*100:.2f}%\n"
            f"Frequentist results:\n"
            f"  P-value                     {self._freq_stats['p_value']:.3f}\n"
            f"  Power                       {self._freq_stats['power']:.3f}"
        )
        # plot details
        if validated_params.stats_title:
            title = f"Distributions:\n{control_report}\n{treatment_report}\n{stats_report_title}"
        else:
            title = f"Distributions:\n{control_report}\n{treatment_report}"
        # plot details
        validated_params.axs[0].set_title(title, loc="left")
        validated_params.axs[0].set_ylabel("Probability\ndensity", loc="bottom")
        validated_params.axs[0].set_xlabel("Parameter Value", loc="left")
        validated_params.axs[0].axvline(
            x=self.control_data.likelihood_prob,
            ymin=0,
            ymax=1,
            color="r",
            linestyle="--",
            linewidth=1.5,
            label="Real control prob",
        )
        validated_params.axs[0].axvline(
            x=self.treatment_data.likelihood_prob,
            ymin=0,
            ymax=1,
            color="b",
            linestyle="--",
            linewidth=1.5,
            label="Real treatment prob",
        )
        validated_params.axs[0].set_xlim(-0.1, 1.1)
        validated_params.axs[0].set_xticks([i / 10 for i in range(0, 11, 1)])
        validated_params.axs[0].legend(bbox_to_anchor=(1.01, 1))
        # plot lift dist distribution
        sns.kdeplot(
            x=self._bayes_stats["lift_dist"],
            color="g",
            ax=validated_params.axs[1],
            linewidth=1.5,
            warn_singular=False,  # zero-variance warning, kde not plot
        )
        # get lift kde
        lift_kde = axs[1].get_lines()[0]
        # get boolean kde within confidence interval
        lift_kde_conf_int_bool = (
            lift_kde.get_xdata() >= self._bayes_stats["lift_cred_lower_limit"]
        ) & (lift_kde.get_xdata() <= self._bayes_stats["lift_cred_upper_limit"])
        # filter x and y data with kde boolean
        lift_kde_conf_int_x = lift_kde.get_xdata()[lift_kde_conf_int_bool]
        lift_kde_conf_int_y = lift_kde.get_ydata()[lift_kde_conf_int_bool]
        axs[1].fill_between(
            x=lift_kde_conf_int_x, y1=0, y2=lift_kde_conf_int_y, color="g"
        )
        # plot details
        validated_params.axs[1].axvline(
            x=self._bayes_stats["lift_rope_lower_limit"],
            ymin=0,
            ymax=1,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="ROPE lower limit",
        )
        validated_params.axs[1].axvline(
            x=self._bayes_stats["lift_rope_upper_limit"],
            ymin=0,
            ymax=1,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="ROPE lower limit",
        )
        lift_report_title = (
            f"\nLift distribution:\n"
            f"  Lift mean         {self._bayes_stats['lift_mean']:.2f}\n"
            f"  Lift interval     [{self._bayes_stats['lift_cred_lower_limit']:.2f}, {self._bayes_stats['lift_cred_upper_limit']:.2f}]\n"
            f"  ROPE interval     [{self._bayes_stats['lift_rope_lower_limit']:.2f}, {self._bayes_stats['lift_rope_upper_limit']:.2f}]"
        )
        validated_params.axs[1].set_title(lift_report_title, loc="left")
        validated_params.axs[1].set_ylabel("Probability\ndensity", loc="bottom")
        validated_params.axs[1].set_xlabel("Parameter Value", loc="left")
        validated_params.axs[1].legend(bbox_to_anchor=(1.01, 1))
