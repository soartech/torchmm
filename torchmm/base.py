"""
This module includes the base Model class used throughout TorCHmm. It also
includes some very basic models for discrete and continuous emissions.
"""
import torch
from torch.distributions import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.gamma import Gamma


class Model(object):
    """
    This is an unsupervised model base class. It specifies the methods that
    should be implemented.
    """

    def init_params_random(self):
        """
        Randomly sets the parameters of the model; used for model fitting.
        """
        raise NotImplementedError("init_params_random method not implemented")

    def log_parameters_prob(self):
        """
        Returns the loglikelihood of the parameter estimates given the prior.
        """
        raise NotImplementedError("log_parameters_prob method not implemented")

    def to(self, device):
        """
        Moves the model's parameters / tensors to the specified device.
        """
        raise NotImplementedError("to method not implemented")

    def sample(self, *args, **kargs):
        """
        Draws a sample from the model. This method might take additional
        arguments for specifying things like how many samples to draw.
        """
        raise NotImplementedError("sample method not implemented")

    def log_prob(self, X):
        """
        Computes the log likelihood of the data X given the model.
        """
        raise NotImplementedError("log_likelihood method not implemented")

    def fit(self, X, *args, **kargs):
        """
        Updates the model parameters to best fit the data X. This might accept
        additional arguments that govern how the model is updated.
        """
        raise NotImplementedError("fit method not implemented")

    def parameters(self):
        """
        Returns the models parameters, as a list of tensors. This is is so
        other models can optimize this model. For example, in the case of
        nested models.
        """
        raise NotImplementedError("parameters method not implemented")

    def set_parameters(self, params):
        """
        Used to set the value of a models parameters
        """
        raise NotImplementedError("set_parameters method not implemented")


class CategoricalModel(Model):

    def __init__(self, probs=None, logits=None, prior=None):
        """
        Accepts a set of probabilites OR logits for the model, NOT both.

        This also accepts a Dirichlet, or counts, prior.
        """
        if probs is not None and logits is not None:
            raise ValueError("Both probs and logits provided; only one should"
                             " be used.")
        elif probs is not None:
            self.logits = probs.float().log()
        elif logits is not None:
            self.logits = logits.float()
        else:
            raise ValueError("Neither probs or logits provided; one must be.")

        if prior is None:
            self.prior = torch.ones_like(self.logits)
        elif prior.shape == logits.shape:
            self.prior = prior.float()
        else:
            raise ValueError("Invalid prior. Ensure shape equals the shape of"
                             " the logits or probs provided.")

        self.device = "cpu"

    def to(self, device):
        self.logits = self.logits.to(device)
        self.prior = self.prior.to(device)
        self.device = device

    def log_parameters_prob(self):
        return Dirichlet(self.prior).log_prob(self.logits.softmax(0))

    def init_params_random(self):
        self.logits = Dirichlet(self.prior).sample().log()

    def sample(self, sample_shape=None):
        """
        Draws n samples from this model.
        """
        if sample_shape is None:
            sample_shape = torch.tensor([1], device=self.device)
        return Categorical(logits=self.logits).sample(sample_shape)

    def log_prob(self, value):
        """
        Returns the loglikelihood of x given the current categorical
        distribution.
        """
        return Categorical(logits=self.logits).log_prob(value)

    def parameters(self):
        """
        Returns the model parameters for optimization.
        """
        yield self.logits

    def set_parameters(self, params):
        """
        Returns the model parameters for optimization.
        """
        self.logits = params[0]

    def fit(self, X):
        """
        Update the logit vector based on the observed counts.

        .. todo::
            Maybe could be modified with weights to support baum welch?
        """
        counts = X.bincount(minlength=self.logits.shape[0]).float()
        prob = (counts + self.prior) / (counts.sum() + self.prior.sum())
        self.logits = prob.log()


class DiagNormalModel(Model):

    def __init__(self, means, precs, means_prior=None, prec_alpha_prior=None,
                 prec_beta_prior=None, n0=None):
        """
        Accepts a set of mean and precisions for each dimension.

        Currently assumes all dimensions are independent.
        """
        if not isinstance(means, torch.Tensor):
            raise ValueError("Means must be a tensor.")
        if not isinstance(precs, torch.Tensor):
            raise ValueError("Covs must be a tensor.")
        if means.shape != precs.shape:
            raise ValueError("Means and covs must have same shape!")

        self.means = means.float()
        self.precs = precs.float()

        if means_prior is None:
            self.means_prior = torch.zeros_like(self.means)
        else:
            self.means_prior = means_prior.float()

        if prec_alpha_prior is None:
            self.prec_alpha_prior = 0.5 * torch.ones_like(self.precs)
        else:
            self.prec_alpha_prior = prec_alpha_prior.float()

        if prec_beta_prior is None:
            self.prec_beta_prior = 0.5 * torch.ones_like(self.precs)
        else:
            self.prec_beta_prior = prec_beta_prior.float()

        if n0 is None:
            self.n0 = torch.tensor(1.)
        else:
            self.n0 = n0.float()

        self.device = "cpu"

    def to(self, device):
        self.means = self.means.to(device)
        self.precs = self.precs.to(device)

        self.means_prior = self.means_prior.to(device)
        self.prec_alpha_prior = self.prec_alpha_prior.to(device)
        self.prec_beta_prior = self.prec_alpha_prior.to(device)
        self.n0 = self.n0.to(device)

        self.device = device

    def init_params_random(self):
        """
        Sample a random parameter configuration from the priors model.
        """
        prec_m = Gamma(self.prec_alpha_prior,
                       self.prec_beta_prior)
        self.precs = prec_m.sample()

        means_m = MultivariateNormal(loc=self.means_prior,
                                     precision_matrix=(self.n0 *
                                                       self.prec_alpha_prior /
                                                       self.prec_beta_prior
                                                       ).diag())
        self.means = means_m.sample()

    def sample(self, sample_shape=None):
        """
        Draws n samples from this model.
        """
        if sample_shape is None:
            sample_shape = torch.tensor([1], device=self.device)
        return MultivariateNormal(
            loc=self.means,
            precision_matrix=self.precs.abs().diag()).sample(sample_shape)

    def log_parameters_prob(self):
        ll = Gamma(self.prec_alpha_prior,
                   self.prec_beta_prior).log_prob(self.precs.abs()).sum()
        ll += MultivariateNormal(
            loc=self.means_prior,
            precision_matrix=(
                self.n0 * self.precs.abs()).diag()).log_prob(self.means)
        return ll

    def log_prob(self, value):
        """
        Returns the loglikelihood of x given the current parameters.
        """
        return MultivariateNormal(
            loc=self.means,
            precision_matrix=self.precs.abs().diag()
        ).log_prob(value)

    def parameters(self):
        """
        Returns the model parameters for optimization.
        """
        yield self.means
        yield self.precs

    def set_parameters(self, params):
        """
        Sets the model parameters.

        .. todo::
            Currently not tested or used.
        """
        print("SETTING", params)
        self.means = params[0]
        self.precs = params[1]

    def fit(self, X):
        """
        Update the logit vector based on the observed counts.

        .. todo::
            - Utilize the priors to do a MAP estimator. See:
                https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/lectures/lecture5.pdf
            - Some of the math seems easier to pull out what I need from pg 8
                here: https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        """
        if X.shape[1] != self.means.shape[0]:
            raise ValueError("Mismatch in number of features.")

        n = X.shape[0]

        # if n == 0, then do nothing.
        if n > 0:
            means = X.mean(0)
            self.means = ((self.n0 * self.means_prior + n * means) /
                          (self.n0 + n))
            alpha = self.prec_alpha_prior + n / 2
            sq_error = (X - means).pow(2)
            beta = (self.prec_beta_prior +
                    0.5 * sq_error.sum(0) +
                    (n * self.n0) * (means - self.means_prior).pow(2) /
                    (2 * (n + self.n0)))
            self.precs = alpha / beta
