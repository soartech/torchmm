import numpy as np
from sklearn.metrics import adjusted_rand_score
import torch

from hmmlearn.hmm import MultinomialHMM

from torchmm.hmm import HiddenMarkovModel
from torchmm.base import CategoricalModel


def fit_hmm_learn(X, n_states):
    samples = np.concatenate(X)
    lengths = [len(x) for x in X]

    hmm_learn_model = MultinomialHMM(n_components=n_states)
    hmm_learn_model.fit(samples, lengths)

    # Label data using hmmlearn model
    return hmm_learn_model.predict(samples, lengths)


def fit_torchmm(X, n_states):
    torch_data = [torch.tensor(x).squeeze() for x in X]
    n_emis = 8
    states = [CategoricalModel(probs=torch.ones(n_emis)/n_emis)
              for _ in range(n_states)]
    torchmm_model = HiddenMarkovModel(states)
    torchmm_model.fit(torch_data, restarts=50, randomize_first=True)

    # Label data using torchmm model
    torchmm_labels, _ = torchmm_model.decode(torch_data)
    return torch.cat(torchmm_labels)


if __name__ == "__main__":

    # Number of states
    n_states = 4

    # Load data
    X = []
    with open('data/biofam.csv', 'r') as fin:
        for row in fin:
            X.append(np.array([[int(e)] for e in row.split(",")]))

    # Create HMM Learn model and fit to data
    hmm_learn_labels_1 = fit_hmm_learn(X, n_states)
    hmm_learn_labels_2 = fit_hmm_learn(X, n_states)

    # Create torchmm model and fit to data
    torchmm_labels_1 = fit_torchmm(X, n_states)
    torchmm_labels_2 = fit_torchmm(X, n_states)

    # Compare models using ARI
    print("Inter HMMLearn ARI=%0.3f" % adjusted_rand_score(hmm_learn_labels_1,
                                                           hmm_learn_labels_2))
    print("Inter Torchmm ARI=%0.3f" % adjusted_rand_score(torchmm_labels_1,
                                                          torchmm_labels_2))
    print("HMMLearn vs Torchmm ARI=%0.3f" %
          adjusted_rand_score(torchmm_labels_1, hmm_learn_labels_1))
