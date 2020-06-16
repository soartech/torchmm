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
        Moves the model's parameters / tensors to the specified pytorch device.
        """
        raise NotImplementedError("to method not implemented")

    def sample(self, *args, **kargs):
        """
        Draws a sample from the model. This method might take additional
        arguments for specifying things like how many samples to draw.

        The shape of the returned object may be dependent on the model.
        """
        raise NotImplementedError("sample method not implemented")

    def log_prob(self, X):
        """
        Returns the log likelihood of the data X given the model.
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
    """
    A model used for representing categorical/discrete outputs from a state in
    the HMM. Uses a categorical distribution and a Dirchlet prior.
    """

    def __init__(self, probs=None, prior=None):
        """
        Accepts a set of probabilites for the model, NOT both. This
        also accepts a Dirichlet, or counts, prior. The prior is also a vector
        with the same shape as probs. If no prior is provided, then it
        defaults to a vector of 1's (i.e., add-one laplace smoothing).
        """
        if probs is None:
            raise ValueError("Probs must be provided.")
        if not torch.isclose(probs.sum(), torch.tensor(1.0)):
            raise ValueError("Probs must sum to 1.")

        self.probs = probs.float()

        if prior is None:
            self.prior = torch.ones_like(self.probs)
        elif prior.shape == probs.shape:
            self.prior = prior.float()
        else:
            raise ValueError("Invalid prior. Ensure shape equals the shape of"
                             " the probs provided.")

        self.device = "cpu"

    def to(self, device):
        """
        Moves the model's parameters / tensors to the specified pytorch device.
        """
        self.probs = self.probs.to(device)
        self.prior = self.prior.to(device)
        self.device = device

    def log_parameters_prob(self):
        """
        Returns the loglikelihood of the parameter estimates given the
        Dirichlet prior.
        """
        return Dirichlet(self.prior).log_prob(self.probs)

    def init_params_random(self):
        """
        Randomly samples the probs/logits using the Dirchlet prior.
        """
        self.probs = Dirichlet(self.prior).sample()

    def sample(self, sample_shape=None):
        """
        Draws samples from this model and returns them in the specified shape.
        """
        if sample_shape is None:
            sample_shape = torch.tensor([1], device=self.device)
        return Categorical(probs=self.probs).sample(sample_shape)

    def log_prob(self, value):
        """
        Returns the loglikelihood of x given the current categorical
        distribution defined by the probs/logits.

        Note that value can have an arbitrary shape and the returned value will
        have the same shape.
        """
        return Categorical(probs=self.probs).log_prob(value)

    def parameters(self):
        """
        Returns the model parameters for optimization.
        """
        yield self.probs

    def set_parameters(self, params):
        """
        Sets the params for the model to the first element of the provided
        params.
        """
        self.probs = params[0]

    def fit(self, X):
        """
        Update the logit vector based on the observed counts using maximum
        likelihood estimation; i.e., it computes the probabilities that
        maximize the data (the mean of the values), converts to and sets the
        corresponding logits.

        .. todo::
            Maybe could be modified with weights to support baum welch?
        """
        counts = X.bincount(minlength=self.probs.shape[0]).float()
        self.probs = (counts + self.prior) / (counts.sum() + self.prior.sum())


class DiagNormalModel(Model):
    """
    A model used for representing multivariate continuous outputs from a state
    in the HMM. Uses a normal distribution with a diagonal covariance matrix;
    i.e., it assumes the features are independent of each other. In essence,
    this assumes that states in the HMM are axis-aligned ellipses.

    The model also uses a Normal-Gamma prior for the mean and covariance matrix
    values.
    """

    def __init__(self, means, precs, means_prior=None, prec_alpha_prior=None,
                 prec_beta_prior=None, n0=None):
        """
        Accepts a set of mean and precisions for each dimension. Means and
        precs must have the same length, corresponding to the number of
        features in each observation.

        The model can accepts a means prior, which imposes a normal prior over
        the estimated means. This prior has the same length as the means/precs
        (a separate prior can be specified for each feature). If no means
        prior is specified, it uses a default prior of 0 (i.e., it pulls the
        estimated means to be closer to zero).

        The model can accept alpha and beta parameters (from a Gamma
        Distribution) that impose a prior on the precision estimates. Both
        alpha and beta must have the same length as the means/precs (a separate
        prior can be specified for each feature). If none are provided, then
        the model assumes the alphas and betas are 1, which yields a gamma
        prior with a mean of 1 and std of 1 over the precision values.

        The model also accepts an n0 value, which specifies how strongly to
        weight the means priors. By default this value is 1, which roughly
        corresponds to having observed 1 previous datapoint with feature values
        equal to the means prior.

        Finally, it is worth mentioning that the means, alpha, beta, and n0
        values jointly define a normal-gamma prior, so they are not independent
        of each other. For example, changing the means or n0 value might impact
        how strongly the alpha and beta values affect the precision estimates.
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
            self.prec_alpha_prior = torch.ones_like(self.precs)
        else:
            self.prec_alpha_prior = prec_alpha_prior.float()

        if prec_beta_prior is None:
            self.prec_beta_prior = torch.ones_like(self.precs)
        else:
            self.prec_beta_prior = prec_beta_prior.float()

        if n0 is None:
            self.n0 = torch.tensor(1.)
        else:
            self.n0 = n0.float()

        self.device = "cpu"

    def to(self, device):
        """
        Moves the model's parameters / tensors to the specified pytorch device.
        """
        self.means = self.means.to(device)
        self.precs = self.precs.to(device)

        self.means_prior = self.means_prior.to(device)
        self.prec_alpha_prior = self.prec_alpha_prior.to(device)
        self.prec_beta_prior = self.prec_alpha_prior.to(device)
        self.n0 = self.n0.to(device)

        self.device = device

    def init_params_random(self):
        """
        Sample a random parameter configuration from the normal-gamma priors
        model.

        For more details on sampling from a normal-gamma distribution see:
            - https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/lectures/lecture5.pdf # noqa: E501
            - https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
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
        Draws n samples from this model and return them in the specified shape.
        """
        if sample_shape is None:
            sample_shape = torch.tensor([1], device=self.device)
        return MultivariateNormal(
            loc=self.means,
            precision_matrix=self.precs.abs().diag()).sample(sample_shape)

    def log_parameters_prob(self):
        """
        Returns the log-likelihood of the parameter estimates given the
        normal-gamma prior.

        For more details computing log-likelihood under this prior see:
            - https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/lectures/lecture5.pdf # noqa: E501
            - https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        """
        ll = Gamma(self.prec_alpha_prior,
                   self.prec_beta_prior).log_prob(self.precs.abs()).sum()
        ll += MultivariateNormal(
            loc=self.means_prior,
            precision_matrix=(
                self.n0 * self.precs.abs()).diag()).log_prob(self.means)
        return ll

    def log_prob(self, value):
        """
        Returns the loglikelihood of the provided values given the current
        parameters.

        Note, this does not include the priors.
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
        Sets the model parameters, the first parameter provided is the means
        and the second is the precisions.

        .. todo::
            Currently not tested or used.
        """
        self.means = params[0]
        self.precs = params[1]

    def fit(self, X):
        """
        Update the means and precisions based on the provided observations
        using an Maximum a posteri (MAP) estimate; i.e., the maximum likelihood
        estimate over both the data and the normal-gamma prior.

        For more details on this calculations see:
            - Utilize the priors to do a MAP estimator. See:
                https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/lectures/lecture5.pdf
            - Some of the math seems easier to pull out what I need from pg 8
                here: https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf

        Note, if X is empty (no observations), then no update is made and the
        means and precisions remain the same.
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
