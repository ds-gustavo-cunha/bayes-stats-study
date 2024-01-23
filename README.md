# BAYES STUDY PROJECT

### PROBLEM

A few times in the past, I was interested in Bayesian statistics. For me, it makes total sense to the way humans think: we start with an initial hypothesis, then we keep on studying and gathering knowledge about the problem and every new information somehow changes the way we thought before. This is more or less the general idea of Bayes formulae (at a very high level, okay?).

With this in mind, I attended some courses about the basics of Bayesian statistics but I was still struggling to get a clear intuition about what is going on behind the scenes. I mean, I'm not a statistician and I really don't want to be one. I'm just a data scientist trying to add new "toys" to my toolbox so I increase my problem-solving ability. With this mindset and getting inspiration from a very good Bayesian AB test course I attended at [Comunidade DS](https://www.comunidadeds.com/), I decided to create this project to play with Bayes statistics and AB testing.

So, the main purpose of this project to to build intuition about Bayes update and AB testing, especially the practical difference between Bayesian and frequentist AB testing.

I can't forget to stress that [MASP — Nosso framework de marketing digital](https://medium.com/loftbr/masp-nosso-framework-de-marketing-digital-3ec46bfc2f96) was another fundamental reference for this project.

### SOLUTION

In simple terms, I want to be able to define statistical distributions and likelihood and check out what will be the update of it. Besides, I want to run an AB test with two chosen distributions and inspect any possible different results a Bayesian and a frequentist test may return

### TASK

Even though I did my best to write a bug-free code, it still needs some review so if you find any bug or point of improvement don't hesitate to let me know.

**Installation**

First, activate your virtual environment. If using `pyenv`, then: 
```bash
pyenv activate your_venv_name
```

Second, install the `bayes_study` package with the following command:
```bash
pip install -e .
```
That's all! We're now ready to build our intuition about Bayes's update

**Usage**

You can use the `notebook/experiments.ipynb` to play with the create package in a notebook environment. The core of the package is two classes: 
- `bayes_study.conjugates.BetaBernoulliConjugate`
    - you can use it to get an intuition about the similarities and differences between prior, likelihood and posterior distribution as well as the relationship among them.
- `bayes_study.ab_test.ABTest`
    - you can use it to get an intuition about the similarities and differences between Bayesian and frequentist statistics of an AB test.

If you want to check animates plots, take a look at `bayes_study.plot_update_animation.py` and `bayes_study.plot_ab_animation.py` and they will present the same plot as in the notebook case but updating the plots over time given the input params. These scripts can be called directly from CLI. 

If you want to call them from the terminal with default params, you can use `make plot_update_animation` and `make plot_ab_animation` (as they are defined in the `Makefile`). If you want to define custom parameters for animated plots, then call the scripts directly passing arguments as CLI arguments.

### RESULT

I learned many things while studying and creating this project. It needs lots of improvements and perhaps some bug fixing but I hope that, if you use it wisely, you can build intuition about the core idea behind the bayesian update.

Feel free to contact me for project feedback, improvements and any other thing you think is relevant.

Thanks for reading and reaching out to this project.




