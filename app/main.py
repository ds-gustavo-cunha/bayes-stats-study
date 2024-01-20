#######################
####### IMPORTS #######
#######################
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta, bernoulli
import streamlit as st
import time


#########################
####### CONSTANTS #######
#########################
# get root of project
PROJECT_ROOT = Path().cwd()
# append project root to path
sys.path.append(os.path.join(PROJECT_ROOT))
# import project lib
from bayes_study.conjugates import BetaBernoulliConjugate
from bayes_study.ab_test import ABTest


######################
####### CONFIG #######
######################
icon = Image.open(os.path.join(PROJECT_ROOT, "img", "bayes_icon.png"))
st.set_page_config(
    page_title="Bayes study experiments",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None,
)

# define app title
st.title("Bayes study: bayes formulae and update intuition")
st.divider()

############################################
####### BAYESIAN FORMULARE INTUITION #######
############################################
# define section title
st.title("Bayes formulae intuition:")


# # define columns to present information
# single_dist_params, single_dist_plot = st.columns(spec=2, gap="small")

# define default params
default_bayes_formulae_dict = dict(
    prior_alpha=1,
    prior_beta=1,
    likelihood_trials=10,
    likelihood_probability=0.7,
    sampling_size=10_000,
)

# check if variable is stored in session
if "bayes_formulae_dict" not in st.session_state:
    # define a dict to store user input
    st.session_state.bayes_formulae_dict = default_bayes_formulae_dict
if "bbc" not in st.session_state:
    # define a dict to store user input
    st.session_state.bbc = None

# # open params input colum
# with single_dist_params:
# define input widgets
left_col, right_col = st.columns(spec=[1, 1], gap="small")
with left_col:
    prior_alpha_formulae_slider = st.slider(
        label="Number of successes on prior:",
        min_value=1,
        max_value=100,
        value=st.session_state.bayes_formulae_dict["prior_alpha"],
        step=1,
        help="None",
    )
with right_col:
    prior_beta_formulae_slider = st.slider(
        label="Number of failures on prior:",
        min_value=1,
        max_value=100,
        value=st.session_state.bayes_formulae_dict["prior_beta"],
        step=1,
        help="None",
    )
left_col, right_col = st.columns(spec=[1, 1], gap="small")
with left_col:
    likelihood_trials_formulae_slider = st.slider(
        label="Number of trials on likelihood:",
        min_value=1,
        max_value=100,
        value=st.session_state.bayes_formulae_dict["likelihood_trials"],
        step=1,
        help="None",
    )
with right_col:
    likelihood_probab_formulae_slider = st.slider(
        label="Probability of successes on likelihood:",
        min_value=0.00,
        max_value=1.00,
        value=st.session_state.bayes_formulae_dict["likelihood_probability"],
        step=0.01,
        help="None",
    )
left_col, right_col = st.columns(spec=[1, 1], gap="small")
with left_col:
    sampling_size_formulae_slider = st.select_slider(
        label="Size of sampling:",
        options=[10, 100, 1_000, 10_000, 100_000, 1_000_000],
        value=st.session_state.bayes_formulae_dict["sampling_size"],
        key="bayes_formulae_intuion_sampling_size",
        help="None",
    )
with right_col:
    update_plot = st.button(
        label="Update plot",
        key="bayes_formulae_intuion_update_plot",
        help=None,
        type="primary",
    )

# create an empty object to populate with plot
empty_plot_space = st.empty()

# check if plot has to be updated
if update_plot:
    # save input variables on session
    st.session_state.bayes_formulae_dict["prior_alpha"] = prior_alpha_formulae_slider
    st.session_state.bayes_formulae_dict["prior_beta"] = prior_beta_formulae_slider
    st.session_state.bayes_formulae_dict[
        "likelihood_probability"
    ] = likelihood_probab_formulae_slider
    st.session_state.bayes_formulae_dict[
        "likelihood_trials"
    ] = likelihood_trials_formulae_slider
    st.session_state.bayes_formulae_dict[
        "sampling_size"
    ] = sampling_size_formulae_slider

    # create a conjugate object
    bbc = BetaBernoulliConjugate(
        prior_alpha=st.session_state.bayes_formulae_dict["prior_alpha"],
        prior_beta=st.session_state.bayes_formulae_dict["prior_beta"],
        likelihood_prob=st.session_state.bayes_formulae_dict["likelihood_probability"],
        likelihood_trials=st.session_state.bayes_formulae_dict["likelihood_trials"],
        sampling_size=st.session_state.bayes_formulae_dict["sampling_size"],
    )
    # save on session
    st.session_state.bbc = bbc
    # update distribution
    st.session_state.bbc.update_distributions()

# if a distribution is available
if st.session_state.bbc is not None:
    # define figure layout
    fig, ax = plt.subplots(figsize=(10, 3))
    # plot distributions
    st.session_state.bbc.plot_dists(fig=fig, ax=ax, st_empty_obj=empty_plot_space)


##########################
####### AB TESTING #######
##########################

# define app title
st.divider()
st.title("Bayes study: AB testing intuition")

# define default params
default_ab_params_dict = dict(
    control_prior_alpha=1,
    control_prior_beta=1,
    treatment_prior_alpha=1,
    treatment_prior_beta=1,
    control_real_success_probab=0.5,
    treatment_real_success_probab=0.6,
    likelihood_trials=10,
    num_posterior_updates=10,
    credible_interval=95,
    rope_limits=3,
    significance_level=95,
    sampling_size=10_000,
)

# check if variable is stored in session
# and save them if not
if "ab_dict" not in st.session_state:
    st.session_state.ab_dict = dict(**default_ab_params_dict)

# define columns to present information
input_col, plot_col = st.columns(spec=2, gap="small")

# open params input colum
with input_col:
    # define column title
    st.header("Treatment and control params")
    # control params
    control_alpha_col, control_beta_col = st.columns(spec=2, gap="small")
    with control_alpha_col:
        # define input widgets
        control_prior_alpha_slider = st.slider(
            label="Number of successes on control prior:",
            min_value=1,
            max_value=100,
            value=int(st.session_state.ab_dict["control_prior_alpha"]),
            step=1,
            help="None",
        )
    with control_beta_col:
        control_prior_beta_slider = st.slider(
            label="Number of failures on control prior:",
            min_value=1,
            max_value=100,
            value=int(st.session_state.ab_dict["control_prior_beta"]),
            step=1,
            help="None",
        )
    # treatment params
    treatment_alpha_col, treatment_beta_col = st.columns(spec=2, gap="small")
    with treatment_alpha_col:
        treatment_prior_alpha_slider = st.slider(
            label="Number of successes on treatment prior:",
            min_value=1,
            max_value=100,
            value=int(st.session_state.ab_dict["treatment_prior_alpha"]),
            step=1,
            help="None",
        )
    with treatment_beta_col:
        treatment_prior_beta_slider = st.slider(
            label="Number of failures on treatment prior:",
            min_value=1,
            max_value=100,
            value=int(st.session_state.ab_dict["treatment_prior_beta"]),
            step=1,
            help="None",
        )
    # real params
    control_real_col, treatment_real_col = st.columns(spec=2, gap="small")
    with control_real_col:
        control_real_success_probab_slider = st.slider(
            label="Real control sucess probability:",
            min_value=0.01,
            max_value=1.00,
            value=st.session_state.ab_dict["control_real_success_probab"],
            step=0.01,
            help="None",
        )
    with treatment_real_col:
        treatment_real_success_probab_slider = st.slider(
            label="Real treatment sucess probability:",
            min_value=0.01,
            max_value=1.00,
            value=st.session_state.ab_dict["treatment_real_success_probab"],
            step=0.01,
            help="None",
        )
    # sampling params
    num_updates_col, likeli_samples_col = st.columns(spec=2, gap="small")
    with num_updates_col:
        ab_num_updates_slider = st.slider(
            label="Number of posterior updates:",
            min_value=1,
            max_value=100,
            value=int(st.session_state.ab_dict["num_posterior_updates"]),
            key="ab_updates",
            step=1,
            help="None",
        )
    with likeli_samples_col:
        ab_likelihood_trials_slider = st.slider(
            label="Number of trials on likelihood:",
            min_value=1,
            max_value=100,
            value=int(st.session_state.ab_dict["likelihood_trials"]),
            key="ab_trials",
            step=1,
            help="None",
        )
    num_updates_col, likeli_samples_col = st.columns(spec=2, gap="small")
    # test params
    rvs_col, credible_col = st.columns(spec=2, gap="small")
    with rvs_col:
        ab_sampling_size_slider = st.select_slider(
            label="Size of sampling for prior and posterior:",
            options=[10, 100, 1_000, 10_000, 100_000, 1_000_000],
            value=st.session_state.ab_dict["sampling_size"],
            key="ab_sampling_size",
            help="None",
        )
    with credible_col:
        credible_interval_slider = st.slider(
            label="Credible interval:",
            min_value=0,
            max_value=100,
            value=int(st.session_state.ab_dict["credible_interval"]),
            step=1,
            help="None",
        )
    significance_col, rope_col = st.columns(spec=2, gap="small")
    with significance_col:
        significance_level_slider = st.slider(
            label="Significance level:",
            min_value=0,
            max_value=100,
            value=int(st.session_state.ab_dict["significance_level"]),
            step=1,
            help="None",
        )
    with rope_col:
        rope_limits_slider = st.slider(
            label="ROPE (region of practical equivalence):",
            min_value=0,
            max_value=100,
            value=int(st.session_state.ab_dict["rope_limits"]),
            step=3,
            help="None",
        )
    # define play button
    update_button = st.button(label="Play", help=None, type="primary")

# create an empty object to populate with plot
empty_plot_space = st.empty()

# check if plot has to be updated
if update_button:
    # save input variables on session
    st.session_state.ab_dict["control_prior_alpha"] = control_prior_alpha_slider
    st.session_state.ab_dict["control_prior_beta"] = control_prior_beta_slider
    st.session_state.ab_dict["treatment_prior_alpha"] = treatment_prior_alpha_slider
    st.session_state.ab_dict["treatment_prior_beta"] = treatment_prior_beta_slider
    st.session_state.ab_dict[
        "control_real_success_probab"
    ] = control_real_success_probab_slider
    st.session_state.ab_dict[
        "treatment_real_success_probab"
    ] = treatment_real_success_probab_slider
    st.session_state.ab_dict["likelihood_trials"] = ab_likelihood_trials_slider
    st.session_state.ab_dict["num_posterior_updates"] = ab_num_updates_slider
    st.session_state.ab_dict["sampling_size"] = ab_sampling_size_slider
    st.session_state.ab_dict["credible_interval"] = credible_interval_slider
    st.session_state.ab_dict["significance_level"] = significance_level_slider
    st.session_state.ab_dict["rope_limits"] = rope_limits_slider

    # create a conjugate objects
    control_bbc = BetaBernoulliConjugate(
        prior_alpha=st.session_state.ab_dict["control_prior_alpha"],
        prior_beta=st.session_state.ab_dict["control_prior_beta"],
        likelihood_prob=st.session_state.ab_dict["control_real_success_probab"],
        likelihood_trials=st.session_state.ab_dict["likelihood_trials"],
        sampling_size=st.session_state.ab_dict["sampling_size"],
    )
    treatment_bbc = BetaBernoulliConjugate(
        prior_alpha=st.session_state.ab_dict["treatment_prior_alpha"],
        prior_beta=st.session_state.ab_dict["treatment_prior_beta"],
        likelihood_prob=st.session_state.ab_dict["treatment_real_success_probab"],
        likelihood_trials=st.session_state.ab_dict["likelihood_trials"],
        sampling_size=st.session_state.ab_dict["sampling_size"],
    )
    # iteration over user input times to update posterior
    for _ in range(st.session_state.ab_dict["num_posterior_updates"]):
        # update objects
        control_bbc.update_distributions()
        treatment_bbc.update_distributions()
    # create a AB test object
    ab_test = ABTest(control_data=control_bbc, treatment_data=treatment_bbc)
    # save ab object
    st.session_state.ab_dict["ab_obj"] = ab_test
    # calculate ab statistics
    st.session_state.ab_dict["ab_obj"].calculate_ab_stats()
    # define figure layout
    fig, ax = plt.subplots(figsize=(10, 3))
    # plot
    st.session_state.ab_dict["ab_obj"].plot_ab_dists(
        fig=fig, ax=ax, st_empty_obj=empty_plot_space, stats_title=True
    )
