black:
	black app bayes_study

plot_ab_animation:
	python3 bayes_study/plot_ab_animation.py

test_plot_ab_params:
	python3 bayes_study/plot_ab_animation.py --control_likelihood_prob 0.3

plot_update_animation:
	python3 bayes_study/plot_update_animation.py

test_plot_update_params:
	python3 bayes_study/plot_update_animation.py --likelihood_prob 0.3