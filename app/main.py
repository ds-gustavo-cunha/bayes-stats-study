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
# import project lib
from bayes_study.conjugates import BetaBernoulliConjugate
from bayes_study.ab_test import ABTest


######################
####### CONFIG #######
######################
st.set_page_config(
    page_title="Bayes product comparison",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None,
)


##########################
####### AB TESTING #######
##########################

# define app title
st.title("Comparing products with Bayesian Statistics")
st.divider()

# define default params
default_ab_params_dict = dict(
    control_product_average_rate=3,
    control_product_num_reviews=1,
    treatment_product_average_rate=3,
    treatment_product_num_reviews=1,
    num_posterior_updates=1,
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
control_product_col, treatment_product_col = st.columns(spec=2, gap="small")

# open params input colum
with control_product_col:
    # define column title
    st.header("Control Product")
    # define input widgets
    control_product_average_rate = st.slider(
        label="Control product averate rate:",
        min_value=0.00,
        max_value=5.00,
        value=3.00,
        step=0.01,
        help="None",
    )
    control_product_num_reviews = st.number_input(
        label="Control product number of reviews:",
        min_value=1,
        max_value=None,
        value=100,
        step=1,
        label_visibility="visible", 
        width="stretch"
    )

# open params input colum
with treatment_product_col:
    # define column title
    st.header("Treatment Product")
    # define input widgets
    treatment_product_average_rate = st.slider(
        label="Treatment product averate rate:",
        min_value=0.00,
        max_value=5.00,
        value=3.00,
        step=0.01,
        help="None",
    )
    treatment_product_num_reviews = st.number_input(
        label="Treatment product number of reviews:",
        min_value=1,
        max_value=None,
        value=100,
        step=1,
        label_visibility="visible", 
        width="stretch"
    )


# define columns to present information
_, advance_stats, _ = st.columns(spec=[1, 2, 1], gap="large")

with advance_stats:
    # Define a toggle for advanced setting
    on = st.toggle("Advanced settings", disabled=False, label_visibility="visible", width="stretch")
    # Whe toggle is expanded
    if on:
        # Define advanced stats settings
        st.session_state.ab_dict["num_posterior_updates"] = st.number_input(
            label="Number of posterior updates",
            min_value=1,
            max_value=None,
            value=1,
            step=1,
            label_visibility="visible", 
            width="stretch"
        )
        st.session_state.ab_dict["credible_interval"] = st.number_input(
            label="Credible interval",
            min_value=1,
            max_value=99,
            value=95,
            step=1,
            label_visibility="visible", 
            width="stretch"
        )
        st.session_state.ab_dict["rope_limits"] = st.number_input(
            label="ROPE",
            min_value=0.1,
            max_value=None,
            value=5.0,
            step=0.1,
            label_visibility="visible", 
            width="stretch"
        )
        st.session_state.ab_dict["significance_level"] = st.number_input(
            label="Significant level",
            min_value=1,
            max_value=99,
            value=5,
            step=1,
            label_visibility="visible", 
            width="stretch"
        )
        st.session_state.ab_dict["sampling_size"] = st.number_input(
            label="Sampling size",
            min_value=1_000,
            max_value=100_000, 
            value=10_000,
            step=1_000,
            label_visibility="visible", 
            width="stretch"
        )


# Define button to compare
debug_flag = st.checkbox("debug session state", value=False, disabled=False, label_visibility="visible", width="content")


# Define button to compare
compare_button = st.button("compare_products", type="primary", width="stretch")

# check if plot has to be updated
if compare_button:
    # Update params
    st.session_state.ab_dict["control_product_average_rate"] = control_product_average_rate
    st.session_state.ab_dict["control_product_num_reviews"] = control_product_num_reviews
    st.session_state.ab_dict["treatment_product_average_rate"] = treatment_product_average_rate
    st.session_state.ab_dict["treatment_product_num_reviews"] = treatment_product_num_reviews

    # create a Beta-Bernoulli conjugate for control product
    control_treatment_productbc = BetaBernoulliConjugate(
        prior_alpha=1,
        prior_beta=1,
        likelihood_prob=(st.session_state.ab_dict["control_product_average_rate"]/5.0),
        likelihood_trials=st.session_state.ab_dict["control_product_num_reviews"],
        sampling_size=st.session_state.ab_dict["sampling_size"] 
    )
    # create a Beta-Bernoulli conjugate for treatment product 
    treatment_product_bbc = BetaBernoulliConjugate(
        prior_alpha=1,
        prior_beta=1,
        likelihood_prob=(st.session_state.ab_dict["treatment_product_average_rate"]/5.0),
        likelihood_trials=st.session_state.ab_dict["treatment_product_num_reviews"],
        sampling_size=st.session_state.ab_dict["sampling_size"]
    )

    # iterate the required number of updates
    for _ in range(st.session_state.ab_dict["num_posterior_updates"]):
        # each time update posterior distributions for both conjugates
        control_treatment_productbc.update_distributions()
        treatment_product_bbc.update_distributions()

    # instanciate a AB testing object
    st.session_state.ab_dict["ab_obj"] = ABTest(
        control_data=control_treatment_productbc, 
        treatment_data=treatment_product_bbc
    )
    # calculate AB testing statistics
    st.session_state.ab_dict["ab_obj"].calculate_ab_stats(
        credible_interval=st.session_state.ab_dict["credible_interval"],
        rope=st.session_state.ab_dict["rope_limits"],
        significance_level=st.session_state.ab_dict["significance_level"]        
    )
    # plot
    fig, axs = plt.subplots(
        ncols=1, nrows=2, constrained_layout=True,
        figsize=(10, 12))
    st.session_state.ab_dict["ab_obj"].plot_ab_dists(
        axs=axs, fig=fig, 
        stats_title=True
    )
    st.pyplot(
        fig=fig,
        clear_figure=None,
        width="content",  # stretch, content
    )

    # Display session state in case of debugging mode
    if debug_flag:
        st.write(st.session_state.ab_dict)