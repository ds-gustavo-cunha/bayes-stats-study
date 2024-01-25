# import required libs
from typing import List, Dict, Union, Any
from pydantic import BaseModel, Field, field_validator
import numpy as np


# pydantic validators
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


def _validate_likelihood_dist(value: Union[List[int], None]) -> Union[List[int], None]:
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
