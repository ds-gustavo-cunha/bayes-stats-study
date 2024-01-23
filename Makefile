black:
	black app bayes_study

plot_ab_animation:
	python3 bayes_study/plot_ab_animation.py

test_plot_params:
	python3 bayes_study/plot_ab_animation.py --control_likelihood_prob 0.3