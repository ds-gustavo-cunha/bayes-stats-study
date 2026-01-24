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
    product_a_average_rate=3,
    product_a_num_reviews=1,
    product_b_average_rate=3,
    product_b_num_reviews=1,
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
product_a_col, product_b_col = st.columns(spec=2, gap="small")

# open params input colum
with product_a_col:
    # define column title
    st.header("Product A")
    # define input widgets
    product_a_average_rate = st.slider(
        label="Product A averate rate:",
        min_value=0.00,
        max_value=5.00,
        value=3.00,
        step=0.01,
        help="None",
    )
    product_a_num_reviews = st.number_input(
        label="Product A number of reviews:",
        min_value=1,
        max_value=None,
        value=100,
        step=1,
        label_visibility="visible", 
        width="stretch"
    )

# open params input colum
with product_b_col:
    # define column title
    st.header("Product B")
    # define input widgets
    product_b_average_rate = st.slider(
        label="Product B averate rate:",
        min_value=0.00,
        max_value=5.00,
        value=3.00,
        step=0.01,
        help="None",
    )
    product_b_num_reviews = st.number_input(
        label="Product B number of reviews:",
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

# create an empty object to populate with plot
empty_plot_space = st.empty()

# Define button to compare
compare_button = st.button("compare_products", type="primary", width="stretch")


# check if plot has to be updated
if compare_button:
    # Update params
    st.session_state.ab_dict["product_a_average_rate"] = product_a_average_rate
    st.session_state.ab_dict["product_a_num_reviews"] = product_a_num_reviews
    st.session_state.ab_dict["product_b_average_rate"] = product_b_average_rate
    st.session_state.ab_dict["product_b_num_reviews"] = product_b_num_reviews

    # create a Beta-Bernoulli conjugate for product A
    product_a_bbc = BetaBernoulliConjugate(
        prior_alpha=1,
        prior_beta=1,
        likelihood_prob=(st.session_state.ab_dict["product_a_average_rate"]/5.0),
        likelihood_trials=st.session_state.ab_dict["product_a_num_reviews"],
        sampling_size=st.session_state.ab_dict["sampling_size"] 
    )
    # create a Beta-Bernoulli conjugate for product B
    product_b_bbc = BetaBernoulliConjugate(
        prior_alpha=1,
        prior_beta=1,
        likelihood_prob=(st.session_state.ab_dict["product_b_average_rate"]/5.0),
        likelihood_trials=st.session_state.ab_dict["product_b_num_reviews"],
        sampling_size=st.session_state.ab_dict["sampling_size"]
    )

    # iterate the required number of updates
    for _ in range(st.session_state.ab_dict["num_posterior_updates"]):
        # each time update posterior distributions for both conjugates
        product_a_bbc.update_distributions()
        product_b_bbc.update_distributions()

    # instanciate a AB testing object
    st.session_state.ab_dict["ab_obj"] = ABTest(
        control_data=product_a_bbc, 
        treatment_data=product_b_bbc
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


# Define button to compare
debug_flag = st.checkbox("debug session state", value=False, disabled=False, label_visibility="visible", width="content")
if debug_flag:
    st.write(st.session_state.ab_dict)