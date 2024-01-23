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
    axs: Any = Field(default=..., description="matplotlib.axes._axes.Axes")
    stats_title: bool = Field(
        default=True,
        description="Boolean to indicate whether to plot stats report on title.",
    )