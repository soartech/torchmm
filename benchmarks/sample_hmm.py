from itertools import chain

import numpy as np
import torch
from hmmlearn.hmm import GaussianHMM

# pip install -U git+https://gitlab+deploy-token-4:J8u_LkbQxeV5tpa95mHr@hq
# -git.soartech.com/chris.maclellan/TorCHmM@master
from torchmm.base import DiagNormalModel
from torchmm.hmm_packed import HiddenMarkovModel
from torchmm.utils import kmeans
# from torchmm.utils import kmeans
from torchmm.utils import pack_list
from torchmm.utils import unpack_list


def get_hmm_params(n_states, n_dist=1):
    startprob = np.random.dirichlet(np.ones(n_states) * 1000., size=1)[0]
    # startprob = np.array([0.6, 0.3, 0.1, 0.0])
    # The transition matrix, note that there are no transitions possible
    # between component 1 and 3
    transmat = np.array(
        [np.random.dirichlet(np.ones(n_states) * 1000., size=1)[0] for _ in
         range(n_states)])
    # The means of each component
    gaussian_n = 2
    means = np.random.rand(n_states, gaussian_n)
    # np.random.rand(n_states).reshape(-1, 1)

    # The covariance of each component
    covars = np.random.rand(n_states, gaussian_n)
    # np.tile(np.identity(gaussian_n), (n_states, 1, 1))
    # np.random.rand(n_states)

    # Build an HMM instance and set parameters
    model = GaussianHMM(n_components=n_states, covariance_type="diag")

    # Instead of fitting it from the data, we directly set the estimated
    # parameters, the means and covariance of the components
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.means_ = means
    model.covars_ = covars
    ###############################################################
    return model
    # Generate samples
    # X, Z = model.sample(500)

    # # Plot the sampled data
    # plt.plot(X[:, 0], X[:, 1], ".-", label="observations", ms=6,
    #          mfc="orange", alpha=0.7)
    #
    # # Indicate the component numbers
    # for i, m in enumerate(means):
    #     plt.text(m[0], m[1], 'Component %i' % (i + 1),
    #              size=17, horizontalalignment='center',
    #              bbox=dict(alpha=.7, facecolor='w'))
    # plt.legend(loc='best')
    # plt.show()


def get_centroids(seqX, n_states):
    X = torch.from_numpy(np.stack(list(chain(*seqX)))).float()
    return kmeans(X, n_states)


def BIC(ll, n_data, n_params):
    # n_params = ((model.logit_T0.shape[0] - 1) +
    #             (model.logit_T.shape[0] * (model.logit_T.shape[1] - 1)) +
    #             (2 * sum([len(s.means) for s in model.states])))
    return np.log(n_data) * n_params - 2 * ll


def fit_hmm_torchmm(seqX, n_states, device="cuda"):
    """
    Function to fit the model and return the BIC
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # amount of data
    n_data = len(list(chain(*seqX)))

    n_features = seqX[0][0].shape[0]

    # get centroids with kmeans++
    centroids = get_centroids(seqX, n_states)

    # build the states
    states = [DiagNormalModel(means=centroids[i],
                              precs=torch.ones(n_features))
              for i in range(n_states)]

    # initial start probs
    T0 = torch.zeros(n_states).softmax(0)

    # transition matrix
    T = torch.zeros((n_states, n_states)).softmax(1)

    # build the HMM
    model = HiddenMarkovModel(states, T0=T0, T=T)
    model.to(device)

    # convert seqs to tensors
    seqX = [torch.stack([torch.from_numpy(e).float().to(device) for e in s])
            for s in seqX]

    # pack sequences
    packed_seqX = pack_list(seqX)

    # fit the hmm
    converged = model.fit(packed_seqX)

    if not converged:
        print('torchmm model did not converge!')

    ll = model.log_prob(packed_seqX).item()

    n_params = ((model.logit_T0.shape[0] - 1) +
                (model.logit_T.shape[0] * (model.logit_T.shape[1] - 1)) +
                (2 * sum([len(s.means) for s in model.states])))
    bic = BIC(ll, n_data, n_params)

    return bic, ll, model


def fit_hmm_hmmlearn(seqs, n_states, hmm=None):
    samples = np.concatenate(seqs)
    lengths = np.array([len(s) for s in seqs])
    if len(samples) < n_states:
        return float('inf'), float('-inf'), None, None
    # assert len(samples) >= n_states
    if hmm is None:
        hmm = GaussianHMM(n_components=n_states)

    hmm.fit(samples, lengths)

    if n_states == 1:
        hmm.startprob_ = np.array([1.0])
    #     hmm.transmat_ = np.array([[1.0]])

    if len(samples) == 1:
        hmm.means_ = samples
        cv = np.zeros((1, len(samples[0]))) + hmm.min_covar
        hmm.covars_ = cv

    ll = hmm.score(samples, lengths)

    n_data = np.sum(lengths)
    n_params = ((hmm.startprob_.shape[0] - 1) +
                (hmm.transmat_.shape[0] * (hmm.transmat_.shape[1] - 1)) +
                (2 * hmm.means_.shape[0] * hmm.means_.shape[1]))

    bic = BIC(ll, n_data, n_params)

    # print("Model with %i states: BIC= %0.2f (ll=%0.2f)" % (n_states, bic,
    # ll))
    return bic, ll, hmm


def predict_torchmm(seqX, model, device="cuda"):
    """
    Given a sequence and a fit model, compute the labels.
    """
    # get gpu device if available.
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # ensure model is on device
    model.to(device)

    # convert to tensore and move to device
    seqX = [torch.stack([torch.from_numpy(e).float().to(device) for e in s])
            for s in seqX]

    # pack, label, and unpack
    packed_seqX = pack_list(seqX)
    packed_labels, _ = model.decode(packed_seqX)
    labels = unpack_list(packed_labels)

    return [e.cpu().data.numpy() for e in labels]


def compare(n_states=5, seq_num=10, seq_len=10, random_state=42):
    ground = get_hmm_params(n_states)
    # model.startprob_
    # model.transmat_
    # model.means_
    # model.covars_
    seqs, states = [], []
    for _ in range(seq_num):
        seqs_, states_ = ground.sample(seq_len, random_state=random_state)
        seqs.append(seqs_)
        states.append(states_)
        random_state += 1

    results = {}
    for use_torchmm in [True, False]:
        model_str = 'torchmm' if use_torchmm else 'hmmlearn'
        if use_torchmm:
            bic, ll, hmm = fit_hmm_torchmm(seqs, n_states)
            # labels = np.concatenate(predict_torchmm(seqs, hmm))
            results['torchmm'] = (bic, ll, hmm)
        else:
            bic, ll, hmm = fit_hmm_hmmlearn(seqs, n_states)
            # labels = hmm.predict(samples, lengths).tolist()
            results['hmmlearn'] = (bic, ll, hmm)
    return results


if __name__ == "__main__":
    # hmm_c = functools.partial(get_hmm_params, 5)
    r = compare()
    print(r)

