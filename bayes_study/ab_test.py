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
import streamlit as st


#####################
####### CLASS #######
class ABTest:
    def __init__(
        self,
        control_data: BetaBernoulliConjugate,
        treatment_data: BetaBernoulliConjugate,
    ):
        # assign input as attributes
        self.control_bbc = control_data
        self.treatment_bbc = treatment_data
        # initialize empty dict for tests
        self.bayes_stats = dict()
        self.freq_stats = dict()

    def calculate_bayesian_stats(self, credible_interval: int = 90) -> Dict:
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
        self.bayes_stats["lift_cred_inter"] = credible_interval
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

        return self.bayes_stats

    def calculate_frequentist_stats(self, significance_level: int = 5) -> Dict:
        # define required variables
        self.freq_stats["control_obs"] = (
            self.control_bbc.posterior_alpha + self.control_bbc.posterior_beta
        )
        self.freq_stats["treatment_obs"] = (
            self.treatment_bbc.posterior_alpha + self.treatment_bbc.posterior_beta
        )
        self.freq_stats["signif_level"] = significance_level
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
        self, credible_interval: int = 90, significance_level: int = 5
    ) -> Dict:
        # calculate bayes and frequentist stats
        self.calculate_bayesian_stats(credible_interval=credible_interval)
        self.calculate_frequentist_stats(significance_level=significance_level)
        self.ab_stats = {**self.bayes_stats, **self.freq_stats}
        # return ab stats except the distributions
        return {k: v for k, v in self.ab_stats.items() if not k.endswith("dist")}

    def plot_ab_dists(self, fig, ax, st_empty_obj=None, stats_title: bool = False):
        # define style to use
        plt.style.use("fivethirtyeight")
        # define figure layout
        fig, ax = plt.subplots()
        # plot treatment and control distributions
        sns.kdeplot(
            x=self.control_bbc.posterior_dist, color="r", label="control", ax=ax
        )
        sns.kdeplot(
            x=self.treatment_bbc.posterior_dist, color="b", label="treatment", ax=ax
        )
        # define title variables
        control_report = f"Control:      Beta({self.control_bbc.posterior_alpha}, {self.control_bbc.posterior_beta})"
        treatment_report = f"Treatment: Beta({self.treatment_bbc.posterior_alpha}, {self.treatment_bbc.posterior_beta})"
        stats_report_title = (
            f"Bayes results:\n"
            f"  Probability of T better C: {self.bayes_stats['probab_t_better_c']*100:.2f}%\n"
            f"  Expected loss: {self.bayes_stats['expected_loss']*100:.2f}%\n"
            f"  Expected uplift: {self.bayes_stats['uplift_t']*100:.2f}%\n"
            f"Frequentist results:\n"
            f"  P-value: {self.freq_stats['p_value']:.3f}\n"
            f"  Power: {self.freq_stats['power']:.3f}"
        )
        # plot details
        if stats_title:
            title = f"{control_report}\n{treatment_report}\n{stats_report_title}"
        else:
            title = f"{control_report}\n{treatment_report}"
        # plot details
        ax.set_title(title, loc="left")
        ax.set_xlabel("Parameter Value", loc="left")
        ax.set_ylabel("Probability Density", loc="top")
        ax.axvline(
            x=self.control_bbc.likelihood_prob,
            ymin=0,
            ymax=1,
            color="r",
            linestyle="--",
            linewidth=1.5,
            label="Real control prob",
        )
        ax.axvline(
            x=self.treatment_bbc.likelihood_prob,
            ymin=0,
            ymax=1,
            color="b",
            linestyle="--",
            linewidth=1.5,
            label="Real treatment prob",
        )
        ax.set_xlim(-0.1, 1.1)
        ax.set_xticks([i / 10 for i in range(0, 11, 1)])
        ax.legend(bbox_to_anchor=(1.01, 1))
        if st_empty_obj is not None:
            st_empty_obj.pyplot(fig=fig, clear_figure=None, use_container_width=False)
        else:
            plt.show()
