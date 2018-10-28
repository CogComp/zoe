# ZOE (Zero-shot Open Entity Typing)
A state of the art system for zero-shot entity fine typing with minimum supervision

## Introduction

This is a demo system for our paper "Zero-Shot Open Entity Typing as Type-Compatible Grounding",
which at the time of publication represents the state-of-the-art of zero-shot entity typing.

The original experiments that produced all the results in the paper
are done with a package written in Java. This is a re-written package solely for
the purpose of demoing the algorithm and validating key results. 

The results may be slightly different with published numbers, due to the randomness in Java's 
HashSet and Python set's iteration order. The difference should be negligible.

This system may take a long time if ran on a large number of new sentences, due to ELMo processing.
We have cached ELMo results for the provided experiments.

The package also contains an online demo, please refer to [Publication Page](http://cogcomp.org/page/publication_view/845)
for more details.

## Usage

### Install the system

#### Prerequisites

* Minimum 20G available disk space and 16G memory. (strict requirement)
* Python 3.X (Mostly tested on 3.5)
* A POSIX OS (Windows not supported)
* Java JDK and Maven
* `virtualenv` if you are installing with script
* `wget` if you are installing with script (Use brew to install it on OSX)
* `unzip` if you are installing with script

#### Install using a one-line command

To make life easier, we provide a simple way to install with `sh install.sh`.

This script does everything mentioned in the next section, plus creating a virtualenv. Use `source venv/bin/activate` to activate.

#### Install manually

See wiki [manual-installation](https://github.com/CogComp/zoe/wiki/Manual-Installation)

### Run the system

Currently you can do the following without changes to the code:
* Run experiment on FIGER test set (randomly sampled as the paper): `python3 main.py figer`
* Run experiment on BBN test set: `python3 main.py bbn`
* Run experiment on the first 1000 Ontonotes_fine test set instances (due to size issue): `python3 main.py ontonotes`

Additionally, you can run server mode that initializes the online demo with `python3 server.py`
However, this requires some additional files that's not provided for download yet.
Please directly contact the authors.

It's generally an expensive operation to run on large numerb of new sentences, but you are welcome to do it.
Please refer to `main.py` and [Engineering Details](https://github.com/CogComp/zoe/wiki/Engineering-Details) 
to see how you can test on your own data. 


## Citation
See the following paper: 
```
@inproceedings{ZKTR18,
    author = {Ben Zhou, Daniel Khashabi, Chen-Tse Tsai and Dan Roth },
    title = {Zero-Shot Open Entity Typing as Type-Compatible Grounding},
    booktitle = {EMNLP},
    year = {2018},
}
```
