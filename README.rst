Introduction
============
Domaincaller is an implementation of the original Directionality Index (DI) based
TAD caller proposed by Dixon et al. [1]_ Instead of original separate scripts for
each stage of the caller, this module provides a convenient command line interface
integrating the whole pipeline, including calculating the DI track, performing HMM
and post-processing. It supports the `.cool <https://github.com/mirnylab/cooler>`_
matrix format, so has low memory requirements when dealing with high resolution data.

Installation
============
First install dependencies using `conda <https://conda.io/miniconda.html>`_::

    conda config --add channels defaults
    conda config --add channels bioconda
    conda config --add channels conda-forge
    conda create -n domaincaller python=3.7.1 cooler=0.8.6 numpy=1.17.2 scipy=1.3.1 pomegranate=0.10.0 networkx=1.11
    conda activate nholoop

Then install the domaincaller using pip::

    pip install domaincaller

Usage
=====
Check the command-line help by ``domaincaller [-h]``.


Citation
========
.. [1] Dixon JR, Selvaraj S, Yue F, Kim A, Li Y, Shen Y, Hu M, Liu JS, Ren B. Topological domains
   in mammalian genomes identified by analysis of chromatin interactions. Nature, 2012,
   doi: 10.1038/nature11082
