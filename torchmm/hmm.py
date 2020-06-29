import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchmm.hmm_packed import HiddenMarkovModel
from torchmm.base import CategoricalModel
from torchmm.utils import unpack_list
from torchmm.utils import pack_list

from typing import List
from typing import Union

from typing import Tuple
from typing import Callable


class HiddenMarkovModel(HiddenMarkovModel):
    """
    A Hidden Markov Model, defined by states (defined by other models that
    determine the likelihood of observation), initial start probabilities, and
    transition probabilities.

    This package wraps the hmm_packed HiddenMarkovModel to provide some
    automattion for packing and unpacking sequences. The general pattern for
    this class is that data is returned in the format it was provided. If
    packed data is provided, then packed results are returned. However, if
    lists of tensors are provided, then lists of tensors are returned.
    """

    def sample(self, n_seq: int, n_obs: int, packed: bool = False) -> Union[
            PackedSequence, List[Tensor]]:
        """
        Draws n_seq samples from the HMM. All sequences will have length
        num_obs. If packed is True, then the model returns a packed list.

        :param n_seq: Number of samples (sequences).
        :param n_obs: Number of observations in each sample (i.e. sequence
            length)
        :param packed: Return a packed list (default False, return
        :returns:  PackedSequence or a list of pytorch tensors corresponding to
            hmm samples.
        """
        packed_obs, packed_states = super().sample(n_seq, n_obs)
        if packed:
            return packed_obs, packed_states

        obs = unpack_list(packed_obs)
        states = unpack_list(packed_states)
        return obs, states

    def filter(self, X: Union[PackedSequence, Tensor, List[Tensor]]) -> Tensor:
        """
        Compute the log posterior distribution over the last state in
        each sequence--given all the data in each sequence.

        Filtering might also be referred to as state estimation.

        :param X: packed sequence or list of tensors containing observation
            data
        :returns: tensor with shape N x S, where N is number of sequences and S
            is the number of states.
        """
        if isinstance(X, PackedSequence):
            return super().filter(X)

        if isinstance(X, torch.Tensor):
            X = [torch.tensor(x) for x in X.tolist()]

        if isinstance(X, list):
            X = pack_list(X)

        return super().filter(X)

    def decode(self, X: Union[PackedSequence, Tensor, List[Tensor]]) -> Tuple[
            List[Tensor], List[Tensor]]:
        """
        Find the most likely state sequences corresponding to each observation
        in X. Note the state assignments within a sequence are not independent
        and this gives the best joint set of state assignments for each
        sequence; i.e., this finds the state assignments for each observation.

        In essence, this finds the state assignments for each observation.

        :param X: sequence/observation data (packed or unpacked)
        :returns: Two lists of tensors. The first contains state labels,
            the second contains the previous state label (i.e., the path).
        """
        if isinstance(X, PackedSequence):
            return super().decode(X)

        if isinstance(X, torch.Tensor):
            X = [torch.tensor(x) for x in X.tolist()]

        if isinstance(X, list):
            X = pack_list(X)

        state_seq_packed, path_ll_packed = super().decode(X)
        return unpack_list(state_seq_packed), unpack_list(path_ll_packed)

    def smooth(self, X: Union[
            PackedSequence, Tensor, List[Tensor]]) -> PackedSequence:
        """
        Compute the smoothed posterior probability over each state for the
        sequences in X. Unlike decode, this computes the best state assignment
        for each observation independent of the other assignments.

        Note, using this to compute the state state labels for the observations
        in a sequence is incorrect! Use decode instead.

        :param X: packed observation data
        :returns: a packed sequence tensors.
        """
        if isinstance(X, PackedSequence):
            return super().smooth(X)

        if isinstance(X, torch.Tensor):
            X = [torch.tensor(x) for x in X.tolist()]

        if isinstance(X, list):
            X = pack_list(X)

        return super().smooth(X)

    def fit(self, X: Union[PackedSequence, Tensor, List[Tensor]],
            max_steps: int = 500, epsilon: float = 1e-3,
            randomize_first: bool = False,
            restarts: int = 10, rand_fun: Callable = None, **kwargs) -> bool:
        """
        .. todo:: Can this docstring be inherited from hmm_packed?

        Learn new model parameters from X using hard expectation maximization
        (viterbi training).

        This has a number of model fitting parameters. max_steps determines the
        maximum number of expectation-maximization steps per fitting iteration.
        epsilon determines the convergence threshold for the
        expectation-maximization (model converges when successive iterations do
        not improve greater than this threshold).

        The expectation-maximization can often get stuck in local maximums, so
        this method supports random restarts (reinitialize and rerun the EM).
        The restarts parameters specifies how many random restarts to perform.
        On each restart the model parameters are randomized using the provided
        rand_fun(hmm, data) function. If no rand_fun is provided, then the
        parameters are sampled using the init_params_random(). If
        randomize_first is True, then the model randomizes the parameters on
        the first iteration, otherwise it uses the current parameter values as
        the first EM start point (the default). When doing random restarts, the
        model will finish with parameters that had the best log-likihood across
        all the restarts.

        The model returns a flag specifying if the best fitting model (across
        the restarts) converged.

        :param X: Sequences/observations
        :param max_steps: Maximum number of iterations to allow viterbi to run
            if it does not converge before then
        :param epsilon: Convergence criteria (log-likelihood delta)
        :param randomize_first: Randomize on the first iteration (restart 0)
        :param restarts: Number of random restarts.
        :param rand_func: Callable F(self, data) for custom randomization
        :param \\**kwargs: arguments for self._viterbi_training
        :returns: Boolean indicating whether or not any of the restarts
            converged
        """
        if isinstance(X, PackedSequence):
            return super().fit(X, max_steps, epsilon, randomize_first,
                               restarts, rand_fun, **kwargs)

        if isinstance(X, torch.Tensor):
            X = [torch.tensor(x) for x in X.tolist()]

        if isinstance(X, list):
            X = pack_list(X)

        return super().fit(X, max_steps, epsilon, randomize_first, restarts,
                           rand_fun, **kwargs)


if __name__ == "__main__":
    T0 = torch.tensor([0.75, 0.25])
    T = torch.tensor([[0.85, 0.15],
                      [0.12, 0.88]])
    s1_orig = torch.tensor([1.0, 0.0])
    s2_orig = torch.tensor([0.0, 1.0])
    s1 = CategoricalModel(probs=s1_orig, prior=torch.zeros_like(s1_orig))
    s2 = CategoricalModel(probs=s2_orig, prior=torch.zeros_like(s2_orig))
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T,
                              T0_prior=torch.tensor([0., 0.]),
                              T_prior=torch.tensor([[0., 0.], [0., 0.]]))
    obs_seq, states = model.sample(10, 10)

    print("First 5 Obersvations of seq 0:  ", obs_seq[0][:5])
    print("First 5 Hidden States of seq 0: ", states[0][:5])
    print()

    T0 = torch.tensor([0.49, 0.51])
    T = torch.tensor([[0.6, 0.4],
                      [0.48, 0.52]])
    s1_orig = torch.tensor([0.9, 0.1])
    s2_orig = torch.tensor([0.2, 0.8])
    s1 = CategoricalModel(probs=s1_orig)
    s2 = CategoricalModel(probs=s2_orig)
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)

    converge = model.fit(obs_seq, max_steps=1, epsilon=1e-4)

    # Not enough samples (only 1) to test
    # assert np.allclose(trans0.data.numpy(), True_pi)
    print("Pi Matrix: ")
    print(model.T0)
    print()

    print("Transition Matrix: ")
    print(model.T)
    print()

    print("Emission Matrix: ")
    for s in model.states:
        print([p for p in s.parameters()])
    print()

    print("Reached Convergence: ")
    print(converge)
    print()

    states_seq, _ = model.decode(obs_seq)

    # state_summary = np.array([model.prob_state_1[i].cpu().numpy() for i in
    #                           range(len(model.prob_state_1))])

    # pred = (1 - state_summary[-2]) > 0.5
    # pred = torch.cat(states_seq, 0).data.numpy()
    # true = np.concatenate(states, 0)
    pred = torch.stack(states_seq)
    true = torch.stack(states)
    error = torch.mean(torch.abs(pred - true).float())
    print("Error: ", error)
    assert error >= 0.9 or error <= 0.1
