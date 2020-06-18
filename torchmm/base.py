"""
This module includes the base Model class used throughout TorCHmm. It also
includes some very basic models for discrete and continuous emissions.
"""
from typing import Union
from typing import Tuple
from typing import List
from typing import Optional

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
        Randomly sets the parameters of the model; used for random restarts
        during model fitting.
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
        Returns the models parameters, as a list of tensors. These are the
        parameters that define the model. These can be passed into the
        set_parameters method.
        """
        raise NotImplementedError("parameters method not implemented")

    def set_parameters(self, params):
        """
        Used to set the value of a models parameters. Params should have an
        identical format to that returned by the parameters() method.
        """
        raise NotImplementedError("set_parameters method not implemented")


class CategoricalModel(Model):
    """
    A model used for representing discrete outputs from a state in the HMM.
    Uses a categorical distribution and a Dirchlet prior. The model emits
    integers from 0-n, where n is the length of the probability tensor used to
    initialize the model.
    """

    def __init__(self, probs: Union[torch.Tensor, int],
                 prior: Union[float, torch.Tensor] = 1,
                 device: str = "cpu") -> None:
        """
        Creates a categorical model that emits values from 0 to n, where n is
        the length of the probs tensor. If probs is an int, then the model
        internally sets probs to a tensor of uniform probabilities of the
        provided length The model also accepts a Dirchlet prior, or emission
        psuedo counts, which are used primarily during model fitting.

        :param probs: either a tensor describing the probability of each
            discrete emission (must sum to 1) or the number of emissions (an
            int), which gets converted to a tensor with uniform probabilites.
        :param prior: a tensor of Dirchlet priors used during model fitting,
            each value essentially counts as having observed at least one prior
            training datapoint with that value. A float can also be
            provided, which gets converted into a tensor with the provided
            value for each entry. If no prior is provided it defaults to a
            tensor of ones (add one laplace smoothing).
        :param device: a string representing the desired pytorch device,
            defaults to cpu.

        >>> model_with_unif_prob = CategoricalModel(3)
        >>> model_with_custom_prob = CategoricalModel(torch.tensor([0.3, 0.7])
        >>> model_with_prior = CategoricalModel(2, prior=torch.tensor([2, 9])
        """
        if isinstance(probs, int):
            probs = torch.ones(probs) / probs

        if not torch.isclose(probs.sum(), torch.tensor(1.0)):
            raise ValueError("Probs must sum to 1.")

        self.probs = probs.float()

        if isinstance(prior, (float, int)):
            self.prior = prior * torch.ones_like(self.probs).float()
        elif prior.shape == probs.shape:
            self.prior = prior.float()
        else:
            raise ValueError("Invalid prior. Ensure shape equals the shape of"
                             " the probs provided.")

        self.to(device)

    def to(self, device: str) -> None:
        """
        Moves the model's parameters / tensors to the specified pytorch device.

        :param device: a string representing the desired pytorch device.
        """
        self.probs = self.probs.to(device)
        self.prior = self.prior.to(device)
        self.device = device

    def log_parameters_prob(self) -> torch.Tensor:
        """
        Returns the loglikelihood of the current parameters given the Dirichlet
        prior.

        >>> CategoricalModel(3).log_parameters_prob()
        tensor(0.6931)
        """
        return Dirichlet(self.prior).log_prob(self.probs)

    def init_params_random(self) -> None:
        """
        Randomly samples and sets model parameters from the Dirchlet prior.
        """
        self.probs = Dirichlet(self.prior).sample()

    def sample(self,
               sample_shape: Optional[Tuple[int]] = None) -> torch.Tensor:
        """
        Draws samples from this model and returns them in a tensor with the
        specified shape.  If no shape is provided, then a single sample is
        returned.

        >>> CategoricalModel(3).sample((3, 5))
        tensor([[0, 0, 2, 0, 2],
                [1, 2, 0, 1, 2],
                [1, 0, 1, 1, 0]])
        """
        if sample_shape is None:
            sample_shape = torch.tensor([1], device=self.device)
        return Categorical(probs=self.probs).sample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Returns the loglikelihood of the provided values given the current
        categorical distribution defined by the probs. Returned tensor will
        have same shape as input and will give the log_probability of each
        value. Note, this does not include any adjustment for the priors.

        :param value: a tensor of possible emissions (int from 0 to n) or
            arbitrary shape.

        >>> CategoricalModel(3).log_prob(torch.tensor([[0, 1, 2], [0, 0, 0]]))
        tensor([[-1.0986, -1.0986, -1.0986],
                [-1.0986, -1.0986, -1.0986]])
        """
        return Categorical(probs=self.probs).log_prob(value)

    def parameters(self) -> List[Union[torch.Tensor]]:
        """
        Returns the model parameters. In this case a list containing a clone of
        the probs tensor is returned.

        >>> CategoricalModel(3).parameters()
        [tensor([0.3333, 0.3333, 0.3333])]
        """
        return [self.probs.clone().detach()]

    def set_parameters(self, params: List[Union[torch.Tensor, list]]) -> None:
        """
        Sets the params for the model to the first element of the provided
        params (format matches that output by parameters()). Provided probs
        must sum to 1.

        >>> CategoricalModel(3).set_parameters([tensor([0.3, 0.5, 0.2])])
        """
        probs = params[0]

        if not torch.isclose(probs.sum(), torch.tensor(1.0)):
            raise ValueError("Probs must sum to 1.")

        self.probs = probs

    def fit(self, X: torch.Tensor) -> None:
        """
        Update the probs based on the observed counts using maximum likelihood
        estimation; i.e., it computes the probabilities that maximize the data,
        which reduces to the counts / total.

        :param X: a 1-D tensor of emissions.

        >>> CategoricalModel(3).fit(torch.tensor([0, 0, 1, 1, 2, 0]))
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
    values. This distribution accepts a means_prior, which is the prior for the
    mean value. It also accepts prec_alpha and prec_beta priors, these are the
    alpha and beta parameters for a gamma prior over the precision of the
    model. Note, the mean and precisions are not independent, the Normal-Gamma
    defines a joint distribution over both.
    """

    def __init__(self, means: Union[torch.Tensor, int],
                 precs: Union[torch.Tensor, float] = 1,
                 means_prior: Union[torch.Tensor, float] = 0,
                 n0: int = 1,
                 prec_alpha_prior: Union[torch.Tensor, float] = 1,
                 prec_beta_prior: Union[torch.Tensor, float] = 1,
                 device: str = "cpu") -> None:
        """
        Accepts a set of mean and precisions for each dimension. Means and
        precs must have the same length, corresponding to the number of
        features in each observation.

        :param means: A tensor of the means for each dimension of the gaussian.
            If an int is provided, then a tensor of zeros will be used that has
            length equal to the provided value.
        :param precs: A tensor of precisions for each dimension of the
            gaussian. If a float is provided, then it gets converted to a
            tensor of floats with the provided value. Defaults to 1.
        :param means_prior: A tensor of priors for the mean of each dimension.
            If a float is provided, then it gets converted to a tensor of
            floats with the provided value. Defaults to 0.
        :param n0: An integer representing the strength of the means prior, it
            roughly correspons to the number of data points that have been
            observed with the value equal to means_prior during model fitting.
            Defaults to 1.
        :param prec_alpha_prior: A tensor of alpha priors for the prec of each
            dimension. If a float is provided, then it gets converted to a
            tensor of floats with the provided value. Defaults to 1.
        :param prec_beta_prior: A tensor of beta priors for the prec of each
            dimension. If a float is provided, then it gets converted to a
            tensor of floats with the provided value. Defaults to 1.
        :param device: a string representing the desired pytorch device,
            defaults to cpu.

        >>> model_with_defaults = DiagNormalModel(2)
        >>> model_custom_means = DiagNormalModel(torch.tensor([5., 2.]))
        """
        if isinstance(means, int):
            means = torch.zeros(means)
        elif not isinstance(means, torch.Tensor):
            raise ValueError("Means must be an int or a tensor.")

        if isinstance(precs, (int, float)):
            precs = precs * torch.ones_like(means)
        elif not isinstance(precs, torch.Tensor):
            raise ValueError("precs must be a tensor or float.")

        if means.shape != precs.shape:
            raise ValueError("means and precs must have same shape!")

        self.means = means.float()
        self.precs = precs.float()

        if isinstance(means_prior, (float, int)):
            self.means_prior = means_prior * torch.ones_like(self.means)
        else:
            self.means_prior = means_prior.float()

        if isinstance(n0, (float, int)):
            self.n0 = n0 * torch.tensor(1.)
        else:
            self.n0 = n0.float()

        if isinstance(prec_alpha_prior, (float, int)):
            self.prec_alpha_prior = prec_alpha_prior * torch.ones_like(
                self.precs)
        else:
            self.prec_alpha_prior = prec_alpha_prior.float()

        if isinstance(prec_beta_prior, (float, int)):
            self.prec_beta_prior = prec_beta_prior * torch.ones_like(
                self.precs)
        else:
            self.prec_beta_prior = prec_beta_prior.float()

        self.to(device)

    def to(self, device) -> None:
        """
        Moves the model's parameters / tensors to the specified pytorch device.

        :param device: a string representing the desired pytorch device.
        """
        self.means = self.means.to(device)
        self.precs = self.precs.to(device)

        self.means_prior = self.means_prior.to(device)
        self.prec_alpha_prior = self.prec_alpha_prior.to(device)
        self.prec_beta_prior = self.prec_alpha_prior.to(device)
        self.n0 = self.n0.to(device)

        self.device = device

    def init_params_random(self) -> None:
        """
        Sample and set parameter values from the normal-gamma priors model.

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

    def sample(self,
               sample_shape: Optional[Tuple[int]] = None) -> torch.Tensor:
        """
        Draws samples from this model and returns them in a tensor with the
        specified shape. If no shape is provided, then a single sample is
        returned. Note, the returned tensor's final dimension will correspond
        to the number of dimensions in the gaussian model.

        >>> DiagNormalModel(2).sample()
        tensor([[-1.2747,  0.9839]])

        >>> DiagNormalModel(2).sample((3, 5))
        tensor([[[ 0.5798,  2.1585],
                 [ 0.6801,  1.2888],
                 [-0.5988, -1.0076],
                 [-0.0123, -0.0339],
                 [-0.1755,  0.4791]],
                [[-0.5034,  0.6856],
                 [ 0.2872, -0.5782],
                 [ 0.5888,  1.0531],
                 [ 1.0995,  0.3306],
                 [-0.2562, -0.4988]],
                [[ 1.0225, -0.6346],
                 [ 0.1509,  0.4840],
                 [ 2.1670, -0.6165],
                 [ 1.5515, -0.8755],
                 [-0.5233, -0.2712]]])
        """
        if sample_shape is None:
            sample_shape = torch.tensor([1], device=self.device)
        return MultivariateNormal(
            loc=self.means,
            precision_matrix=self.precs.abs().diag()).sample(sample_shape)

    def log_parameters_prob(self) -> torch.Tensor:
        """
        Returns the log-likelihood of the parameter estimates given the
        normal-gamma prior.

        For more details computing log-likelihood under this prior see:
            - https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/lectures/lecture5.pdf # noqa: E501
            - https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf

        >>> DiagNormalModel(2).log_parameters_prob()
        tensor(-3.8379)
        """
        ll = Gamma(self.prec_alpha_prior,
                   self.prec_beta_prior).log_prob(self.precs.abs()).sum()
        ll += MultivariateNormal(
            loc=self.means_prior,
            precision_matrix=(
                self.n0 * self.precs.abs()).diag()).log_prob(self.means)
        return ll

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Returns the loglikelihood of the provided values given the current
        gaussian distribution defined by the model parameters. Returned tensor
        will have same shape as input minus the final dimension, which
        corresponds to the number of gaussian features. Each entry is the
        log_probability of each emission. Note, this does not include any
        adjustment for the priors.

        :param value: A tensor of possible emissions of arbitrary shape; the
            last dimension corresponds to the number of features in the
            Gaussian.

        >>> DiagNormalModel(2).log_prob(torch.tensor([[-1.2,  0.9],
        >>>                                           [-2.3, 1.1]]))
        tensor([-2.9629, -5.0879])
        """
        return MultivariateNormal(
            loc=self.means,
            precision_matrix=self.precs.abs().diag()
        ).log_prob(value)

    def parameters(self) -> List[Union[torch.Tensor]]:
        """
        Returns the model parameters. In this case a list containing a clone of
        the mean and prec tensors.

        >>> DiagNormalModel(2).parameters()
        [tensor([0., 0.]), tensor([1., 1.])]
        """
        return [self.means.clone().detach(), self.precs.clone().detach()]

    def set_parameters(self, params: List[Union[torch.Tensor, list]]) -> None:
        """
        Sets the params for the model (format matches that output by
        parameters()). In this case a list containing the means and prec
        tensors.

        >>> DiagNormalModel(2).set_parameters([tensor([0., 0.]),
        >>>                                    tensor([1., 1.])])
        """
        self.means = params[0]
        self.precs = params[1]

    def fit(self, X: torch.Tensor) -> None:
        """
        Update the means and precisions based on the provided observations
        using an Maximum a posteri (MAP) estimate; i.e., the maximum likelihood
        estimate over both the data and the normal-gamma prior.

        For more details on this calculations see:
            - Utilize the priors to do a MAP estimator. See:
                https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/lectures/lecture5.pdf
            - Some of the math seems easier to pull out what I need from pg 8
                here: https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf

        Note, if X is empty (no observations), then the means and precisions
        get sent entirely based on the priors.

        :param X: a N x F tensor of emissions, where N is the number of
        emissions and F is the number of features in the Gaussian.

        >>> DiagNormalModel(2).fit(torch.tensor([[ 0.5798,  2.1585],
        >>>                                      [ 0.6801,  1.2888],
        >>>                                      [-0.5988, -1.0076],
        >>>                                      [-0.0123, -0.0339],
        >>>                                      [-0.1755,  0.4791]]))
        """
        if X.shape[1] != self.means.shape[0]:
            raise ValueError("Mismatch in number of features.")

        n = X.shape[0]

        if n == 0:
            self.means = self.means_prior
            self.precs = self.prec_alpha_prior / self.prec_beta_prior
        elif n > 0:
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
