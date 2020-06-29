.. image:: https://github.com/soartech/torchmm/blob/master/docs/logo/TorCHmm.png?raw=true
   :height: 64 px
   :width: 64 px
   :alt: TorCHmM Logo
   :align: left

TorCHmM
=======

.. image:: https://hq-git.soartech.com/chris.maclellan/hmm_torch/badges/master/pipeline.svg
     :target: https://hq-git.soartech.com/chris.maclellan/hmm_torch/commits/master
     :alt: Pipeline Status

.. image:: https://hq-git.soartech.com/chris.maclellan/hmm_torch/badges/master/coverage.svg
     :target: https://chris.maclellan.hq-git.soartech.com/TorCHmM/coverage/
     :alt: Coverage Report

A PyTorch implementation of various hidden Markov model inference and learning
algorithms. The library's primary purpose is to provide a means of fitting
hidden Markov models using GPU hardware. The library provides a modular
interface for fitting HMM's with custom emission models; it has built-in
emission models for discrete and Gaussian outputs. The library also supports
regularization, the ability to fit the model with multiple random restarts, and
supports PyTorch's packed list input represention (so sequences can having
varying length).

API documentation can be found at: https://chris.maclellan.hq-git.soartech.com/TorCHmM
If this link is unavailable, then the documentation can be built locally. To
build a local copy of the documentation, go to the `docs` folder and run the
commands `pip install -r doc-requirements.txt` and `make html` to build an HTML
copy of the docs. These docs can then be accessed locadlly at `_build/html/`.

The package can be installed via pip directly from git. To do this run the following command:

    pip install -U git+https://<GIT URL>@master

substitute the appropriate git url in the command above.

Once the package has been installed, examples of how to use it can be found in
the `benchmarks` folder. 
