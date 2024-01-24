#######################
####### IMPORTS #######
from typing import List, Dict, Union
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
class ABTest:
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
        # sanity check
        if (not isinstance(control_data, BetaBernoulliConjugate)) or (
            not isinstance(treatment_data, BetaBernoulliConjugate)
        ):
            raise ValueError(
                "`control_data` and `treatment_data` must both be "
                "bayes_study.conjugates.BetaBernoulliConjugate objects"
            )
        # assign input as attributes
        self.control_bbc = control_data
        self.treatment_bbc = treatment_data
        # initialize empty dict for tests
        self.bayes_stats = dict()
        self.freq_stats = dict()

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
        self.bayes_stats["probab_t_better_c"] = np.mean(
            self.treatment_bbc.posterior_dist > self.control_bbc.posterior_dist
        )
        # calculate expected loss:
        # expected loss if choose treatment and treatment is worse
        self.bayes_stats["expected_loss"] = np.mean(
            (self.control_bbc.posterior_dist - self.treatment_bbc.posterior_dist)
            / self.treatment_bbc.posterior_dist
        )
        # Calculate uplift:
        # expected improvement if choose treatment and treatment is better
        self.bayes_stats["uplift_t"] = np.mean(
            (self.treatment_bbc.posterior_dist - self.control_bbc.posterior_dist)
            / self.control_bbc.posterior_dist
        )

        # sample lift from distribution
        self.bayes_stats["lift_dist"] = (
            self.treatment_bbc.posterior_dist / self.control_bbc.posterior_dist - 1
        )
        # define credible interval boundaries
        self.bayes_stats["lift_cred_inter"] = validated_params.credible_interval
        tail_limits = (100 - self.bayes_stats["lift_cred_inter"]) / 2
        self.bayes_stats["lift_cred_lower_q"] = tail_limits / 100
        self.bayes_stats["lift_cred_upper_q"] = (
            1 - self.bayes_stats["lift_cred_lower_q"]
        )
        # calculate lift difference and credible interval
        self.bayes_stats["lift_mean"] = np.mean(self.bayes_stats["lift_dist"])
        # calculate lift credible interval
        (
            self.bayes_stats["lift_cred_lower_limit"],
            self.bayes_stats["lift_cred_upper_limit"],
        ) = np.quantile(
            a=self.bayes_stats["lift_dist"],
            q=[
                self.bayes_stats["lift_cred_lower_q"],
                self.bayes_stats["lift_cred_upper_q"],
            ],
        )
        # define lower and upper ROPE limits
        self.bayes_stats["rope"] = validated_params.rope / 100
        (
            self.bayes_stats["lift_rope_lower_limit"],
            self.bayes_stats["lift_rope_upper_limit"],
        ) = (-self.bayes_stats["rope"], self.bayes_stats["rope"])

        return self.bayes_stats

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
        self.freq_stats["control_obs"] = (
            self.control_bbc.posterior_alpha + self.control_bbc.posterior_beta
        )
        self.freq_stats["treatment_obs"] = (
            self.treatment_bbc.posterior_alpha + self.treatment_bbc.posterior_beta
        )
        self.freq_stats["signif_level"] = validated_params.significance_level
        # Perform z-test for proportions
        self.freq_stats["z_stat"], self.freq_stats["p_value"] = proportions_ztest(
            count=[
                self.control_bbc.posterior_alpha,
                self.treatment_bbc.posterior_alpha,
            ],
            nobs=[self.freq_stats["control_obs"], self.freq_stats["treatment_obs"]],
            alternative="two-sided",
        )
        # Calculate the observed proportion for each group
        observed_proportion_c = (
            self.control_bbc.posterior_alpha / self.freq_stats["control_obs"]
        )
        observed_proportion_t = (
            self.treatment_bbc.posterior_alpha / self.freq_stats["treatment_obs"]
        )
        # Calculate the effect size (difference in proportions)
        self.freq_stats["effect_size"] = observed_proportion_t - observed_proportion_c
        # Calculate the power
        self.freq_stats["power"] = zt_ind_solve_power(
            effect_size=self.freq_stats["effect_size"],
            nobs1=self.freq_stats["control_obs"],
            alpha=self.freq_stats["signif_level"] / 100,
            power=None,  # what needs to be calculated
            ratio=1,
            alternative="two-sided",
        )

        return self.freq_stats

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
        self.ab_stats = {**self.bayes_stats, **self.freq_stats}
        # return ab stats except the distributions
        return {k: v for k, v in self.ab_stats.items() if not k.endswith("dist")}

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
            x=self.control_bbc.posterior_dist,
            color="r",
            label="control",
            ax=validated_params.axs[0],
        )
        sns.kdeplot(
            x=self.treatment_bbc.posterior_dist,
            color="b",
            label="treatment",
            ax=validated_params.axs[0],
        )
        # define title variables
        control_report = f"  Control                     Beta({self.control_bbc.posterior_alpha}, {self.control_bbc.posterior_beta})"
        treatment_report = f"  Treatment                   Beta({self.treatment_bbc.posterior_alpha}, {self.treatment_bbc.posterior_beta})"
        stats_report_title = (
            f"Bayes results:\n"
            f"  Probability of T better C   {self.bayes_stats['probab_t_better_c']*100:.2f}%\n"
            f"  Expected loss               {self.bayes_stats['expected_loss']*100:.2f}%\n"
            f"  Expected uplift             {self.bayes_stats['uplift_t']*100:.2f}%\n"
            f"Frequentist results:\n"
            f"  P-value                     {self.freq_stats['p_value']:.3f}\n"
            f"  Power                       {self.freq_stats['power']:.3f}"
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
            x=self.control_bbc.likelihood_prob,
            ymin=0,
            ymax=1,
            color="r",
            linestyle="--",
            linewidth=1.5,
            label="Real control prob",
        )
        validated_params.axs[0].axvline(
            x=self.treatment_bbc.likelihood_prob,
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
            x=self.bayes_stats["lift_dist"],
            color="g",
            ax=validated_params.axs[1],
            linewidth=1.5,
        )
        # get lift kde
        lift_kde = axs[1].get_lines()[0]
        # get boolean kde within confidence interval
        lift_kde_conf_int_bool = (
            lift_kde.get_xdata() >= self.bayes_stats["lift_cred_lower_limit"]
        ) & (lift_kde.get_xdata() <= self.bayes_stats["lift_cred_upper_limit"])
        # filter x and y data with kde boolean
        lift_kde_conf_int_x = lift_kde.get_xdata()[lift_kde_conf_int_bool]
        lift_kde_conf_int_y = lift_kde.get_ydata()[lift_kde_conf_int_bool]
        axs[1].fill_between(
            x=lift_kde_conf_int_x, y1=0, y2=lift_kde_conf_int_y, color="g"
        )
        # plot details
        validated_params.axs[1].axvline(
            x=self.bayes_stats["lift_rope_lower_limit"],
            ymin=0,
            ymax=1,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="ROPE lower limit",
        )
        validated_params.axs[1].axvline(
            x=self.bayes_stats["lift_rope_upper_limit"],
            ymin=0,
            ymax=1,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="ROPE lower limit",
        )
        lift_report_title = (
            f"\nLift distribution:\n"
            f"  Lift mean         {self.bayes_stats['lift_mean']:.2f}\n"
            f"  Lift interval     [{self.bayes_stats['lift_cred_lower_limit']:.2f}, {self.bayes_stats['lift_cred_upper_limit']:.2f}]\n"
            f"  ROPE interval     [{self.bayes_stats['lift_rope_lower_limit']:.2f}, {self.bayes_stats['lift_rope_upper_limit']:.2f}]"
        )
        validated_params.axs[1].set_title(lift_report_title, loc="left")
        validated_params.axs[1].set_ylabel("Probability\ndensity", loc="bottom")
        validated_params.axs[1].set_xlabel("Parameter Value", loc="left")
        validated_params.axs[1].legend(bbox_to_anchor=(1.01, 1))
        plt.show()
