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
