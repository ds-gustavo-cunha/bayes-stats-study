# import required libs
from typing import List, Dict, Union, Any
from pydantic import BaseModel, Field, field_validator
import numpy as np


# pydantic validators
class BetaBernoulliConjugateParams(BaseModel, extra="forbid"):
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
            "Probability of the Bernoulli trials of the " "likelihood distribution."
        ),
    )
    likelihood_trials: Union[int, None] = Field(
        default=None,
        ge=1,
        description=("Number of Bernoulli trials of the " "likelihood distribution."),
    )
    sampling_size: int = Field(
        default=10_000,
        ge=10,
        description=(
            "Size of the `rvs` sampling that will be used on "
            "Beta and Bernoulli distributions."
        ),
    )

    @field_validator(__field="likelihood_dist")
    @classmethod
    def validate_likelihood_dist(cls, value: Union[List, None]):
        # if value is not None
        if value is not None:
            # if not a list
            if not isinstance(value, List):
                raise ValueError("`likelihood_dist` must be a list.")
            # iterate over items
            for i in value:
                # if item not 0 or 1
                # as it is supposed to be the data
                # from a bernoulli distribution
                if i not in [0, 1]:
                    raise ValueError(
                        "Items of `likelihood_dist` must be integers: "
                        " either 0s or 1s. "
                        f"Found `{i}`."
                    )
        return value


class BetaBernoulliConjugatePlotDists(BaseModel, extra="forbid"):
    fig: Any = Field(default=..., description="matplotlib.figure.Figure to plot.")
    ax: Any = Field(default=..., description="matplotlib.axes._axes.Axes to plot.")

    plot_prior: bool = Field(
        default=True,
        description="Boolean to indicate whether to plot prior distribution or not.",
    )
    plot_posterior: bool = Field(
        default=True,
        description="Boolean to indicate whether to plot posterior distribution or not.",
    )
