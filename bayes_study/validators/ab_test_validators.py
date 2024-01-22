# import required libs
from typing import List, Dict, Union, Any
from pydantic import BaseModel, Field, field_validator
from bayes_study.conjugates import BetaBernoulliConjugateParams, BetaBernoulliConjugate
import numpy as np


class CalculateBayesianStatsParams(BaseModel, extra="forbid"):
    credible_interval: int = Field(
        default=95,
        ge=0,
        le=100,
        description="Credible interval for ABTest.calculate_bayesian_stats method.",
    )
    rope: int = Field(
        default=3,
        ge=0,
        description="Credible interval for ABTest.calculate_bayesian_stats method.",
    )


class CalculateFrequentistStatsParams(BaseModel, extra="forbid"):
    significance_level: int = Field(
        default=5,
        ge=0,
        le=100,
        description="Significance level for ABTest.calculate_frequentist_stats method.",
    )


class PlotAbDistsParams(BaseModel, extra="forbid"):
    fig: Any = Field(default=..., description="matplotlib.figure.Figure")
    ax: Any = Field(default=..., description="matplotlib.axes._axes.Axes")
    st_empty_obj: Union[Any, None] = Field(
        default=None, description="streamlit.delta_generator.DeltaGenerator"
    )
    stats_title: bool = Field(
        default=True,
        description="Boolean to indicate whether to plot stats report on title.",
    )


# # pydantic validators
# class ABTestParams(BaseModel, extra="forbid", arbitrary_types_allowed=True):
#     # control_data: BetaBernoulliConjugateParams = Field(
#     #     default=...,
#     #     description="BetaBernoulliConjugate object with control data."
#     # )
#     control_data: BetaBernoulliConjugate = Field(
#         default=...,
#         description="BetaBernoulliConjugate object with control data."
#     )

#     # treatment_data:


# # control_data: BetaBernoulliConjugateParams = Field(
# #     default=...,
# #     description="BetaBernoulliConjugate object with control data."
# # )
# control_data: BetaBernoulliConjugate = Field(
#     default=...,
#     description="BetaBernoulliConjugate object with control data."
# )

# # treatment_data:


#     prior_alpha: int = Field(
#         default=..., ge=1,
#         description="Alpha parameter of prior beta distribution."
#     )
#     prior_beta: int = Field(
#         default=..., ge=1,
#         description="Beta parameter of prior beta distribution."
#     )
#     likelihood_dist: Union[List[int], None] = Field(
#         default=None,
#         description="Numpy array with Bernoulli trials."
#     )
#     likelihood_prob: Union[float, None] = Field(
#         default=None, ge=0.0, lt=1.0,
#         description="Probability of success on likelihood Bernoulli distribution."
#     )
#     likelihood_trials: Union[int, None] = Field(
#         default=None, ge=1,
#         description="Number of trials of likelihood Bernoulli distribution."
#     )
#     sampling_size: int = Field(
#         default=10_000, ge=10,
#         description="Sampling size for `rvs` sampling method of distributions."
#     )

#     @field_validator(__field="likelihood_dist")
#     @classmethod
#     def validate_likelihood_dist(cls, value:Union[List, None]):
#         # if value is not None
#         if (value is not None):
#             # if not a list
#             if not isinstance(value, List):
#                 raise ValueError("`likelihood_dist` must be a list.")
#             # iterate over items
#             for i in value:
#                 # if item not 0 or 1
#                 # as it is supposed to be the data
#                 # from a bernoulli distribution
#                 if i not in [0, 1]:
#                     raise ValueError(
#                         "Items of `likelihood_dist` must be integers: "
#                         " either 0s or 1s. "
#                         f"Found `{i}`."
#                         )
#         return np.array(value)