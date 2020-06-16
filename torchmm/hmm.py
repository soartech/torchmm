import torch
from torch.nn.utils.rnn import PackedSequence

from torchmm.hmm_packed import HiddenMarkovModel
from torchmm.base import CategoricalModel
from torchmm.utils import unpack_list
from torchmm.utils import pack_list
from torchmm.utils import kmeans_init
from torchmm.utils import kmeans


def kpp_rand(hmm, X):
    hmm.init_params_random()
    n_states = len(hmm.states)
    X = torch.stack(unpack_list(X))
    centroids = kmeans_init(X.squeeze(), n_states)

    for s_idx, s in enumerate(hmm.states):
        s.means = centroids[s_idx]


def kmeans_rand(hmm, X):
    hmm.init_params_random()
    n_states = len(hmm.states)
    X = torch.stack(unpack_list(X))
    centroids = kmeans(X.squeeze(), n_states)

    for s_idx, s in enumerate(hmm.states):
        s.means = centroids[s_idx]


class HiddenMarkovModel(HiddenMarkovModel):
    """
    A Hidden Markov Model, defined by states (defined by other models that
    determine the likelihood of observation), initial start probabilities, and
    transition probabilities.

    Note, this model requires that all observation sequences are the same
    length. If the sequences have different lengths, then look at the Hidden
    Markov Model from the hmm_packed module.

    .. todo::
        Consider removing this model and only supporting packed sequences.
    """

    def sample(self, n_seq, n_obs, packed=False):
        """
        Draws n_seq samples from the HMM. All sequences will have length
        num_obs.

        :param n_seq: Number of sequences in generated sample
        :type n_seq: int
        :param n_obs: Number of observations per sequence in generated sample
        :type n_obs: int
        :returns: two tensors of shape n_seq x n_obs. The first contains
            observations of length F, where F is the number of emissions
            defined by the HMM's state models. The second contains state
            labels.
        """
        packed_obs, packed_states = super().sample(n_seq, n_obs)
        if packed:
            return packed_obs, packed_states

        obs = unpack_list(packed_obs)
        states = unpack_list(packed_states)
        return obs, states

    def filter(self, X):
        """
        Compute the log posterior distribution over the last state in
        each sequence--given all the data in each sequence.

        Filtering might also be referred to as state estimation.

        :param X: sequence/observation data
        :type X: tensor with shape N x O x F, where N is number of sequences, O
            is number of observations, and F is number of emission/features
        :returns: tensor with shape N x S, where S is the number of states
        """
        if isinstance(X, PackedSequence):
            return super().filter(X)

        if isinstance(X, torch.Tensor):
            X = [torch.tensor(x) for x in X.tolist()]

        if isinstance(X, list):
            X = pack_list(X)

        return super().filter(X)

    def decode(self, X):
        """
        Find the most likely state sequences corresponding to each observation
        in X. Note the state assignments within a sequence are not independent
        and this gives the best joint set of state assignments for each
        sequence.

        This essentially finds the state assignments for each observation.

        :param X: sequence/observation data
        :type X: tensor with shape N x O x F, where N is number of sequences, O
            is number of observations, and F is number of emission/features
        :returns: two tensors, first is N x O and contains the state labels,
            the second is N x O and contains the previous state label
        """
        if isinstance(X, PackedSequence):
            return super().decode(X)

        if isinstance(X, torch.Tensor):
            X = [torch.tensor(x) for x in X.tolist()]

        if isinstance(X, list):
            X = pack_list(X)

        state_seq_packed, path_ll_packed = super().decode(X)
        return unpack_list(state_seq_packed), unpack_list(path_ll_packed)

    def smooth(self, X):
        """
        Compute the smoothed posterior probability over each state for the
        sequences in X. Unlike decode, this computes the best state assignment
        for each observation independent of the other assignments.

        Note, using this to compute the state state labels for the observations
        in a sequence is incorrect! Use decode instead.

        :param X: sequence/observation data
        :type X: tensor with shape N x O x F, where N is number of sequences, O
            is number of observations, and F is number of emission/features
        :returns: N x O x S
        """
        if isinstance(X, PackedSequence):
            return super().smooth(X)

        if isinstance(X, torch.Tensor):
            X = [torch.tensor(x) for x in X.tolist()]

        if isinstance(X, list):
            X = pack_list(X)

        return super().smooth(X)

    def fit(self, X, **kwargs):
        """
        Learn new model parameters from X using the specified alg.

        :param X: sequence/observation data
        :type X: tensor with shape N x O x F, where N is number of sequences, O
            is number of observations, and F is number of emission/features
        :returns: None
        """
        if isinstance(X, PackedSequence):
            return super().fit(X, **kwargs)

        if isinstance(X, torch.Tensor):
            X = [torch.tensor(x) for x in X.tolist()]

        if isinstance(X, list):
            X = pack_list(X)

        return super().fit(X, **kwargs)


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
    # print(obs_seq[:, 0].sum())

    print("First 5 Obersvations of seq 0:  ", obs_seq[0][:5])
    print("First 5 Hidden States of seq 0: ", states[0][:5])

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

    print("Transition Matrix: ")
    print(model.T)
    # assert np.allclose(transition.exp().data.numpy(), True_T, atol=0.1)
    print()
    print("Emission Matrix: ")
    for s in model.states:
        print([p for p in s.parameters()])
    # assert np.allclose(emission.exp().data.numpy(), True_E, atol=0.1)
    print()
    print("Reached Convergence: ")
    print(converge)

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
