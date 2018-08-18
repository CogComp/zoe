# ZOE
A state of the art system for zero-shot entity fine typing with minimum supervision

## Introduction

This is a demo system for our paper "Zero-Shot Open Entity Typing as Type-Compatible Grounding",
which at the time of publication represents the state-of-the-art of zero-shot entity typing.

The original experiments that produced all the results in the paper
are done with a package written in Java. This is a re-written package 
that contains the same core, without experimental code. It's solely for
the purpose of demoing the algorithm and validating key results. 

The results may slightly differ from published numbers, due to the randomness in Java's 
HashSet iteration order. The difference should be within 0.5%.

A major flaw of this re-written demo is the speed. It's much slower comparing to it's
original version in Java, due to the usage of naive data structures like Python lists.
Re-written with modern packages written in Cython like numpy may hugely improve it.

## Usage

### Install the system

#### Prerequisites

* Python 3.X (Mostly tested on 3.5)
* A POSIX OS (Windows not tested)
* `virtualenv` if you are installing with script (check if `virtualenv` command works)
* `wget` if you are installing with script (Use brew to install it on OSX)
* `unzip` if you are installing with script

#### Install using a shell script

To make everyone's life easier, we have provided a simple way for install, simply run `sh install.sh`.

This script does everything mentioned in the next section, plus creating a virtualenv. Use `source venv/bin/activate` to activate.

#### Install manually

Generally it's recommended to create a Python3 virtualenv and work under it.

You need to first install AllenAI's bilm-tf package by running `python3 setup.py install` in ./bilm-tf directory

Then install requirements by `pip3 install -r requirements.txt` in project root

Then you need to download all the data/model files. There are two steps in this:
* in bilm-tf/, download [model.zip](http://cogcomp.org/Data/ccgPapersData/xzhou45/zoe/model.zip), and uncompress
* project root, download [data.zip](http://cogcomp.org/Data/ccgPapersData/xzhou45/zoe/data.zip), and uncompress

Then check if all files are here by `python3 scripts.py CHECKFILES` or `python3 scripts.py CHECKFILES figer`
in order to check figer caches etc.

### Run the system

Currently you can do the following:
* Run experiment on FIGER test set (randomly sampled as the paper): `python3 main.py figer` (note this usually takes 2 hours)

Experiments on much larger datasets are not available yet due to efficiency reasons (see Introduction),
 but we are working on it.

However, you can still run on random sentences of your choice.
Please refer to `main.py` to see how you can test on your own data. 
However, note that it usually takes a long time since ELMo processing is a very expensive operation.

## Engineering details

### Structure

The package is composed with 

* A slightly modified ELMo source code, see [bilm-tf](https://github.com/allenai/bilm-tf)
* A main library `zoe_utils.py`
* A executor `main.py`
* A script helper `script.py` 

### zoe_utils.py

This is the main library file which contains the core logic.

It has 4 main component Classes:

#### `EsaProcessor`

Supports all operations related to ESA and its data files. 

A main entrance is `EsaProcessor.get_candidates` which given a sentence, returns 
the top `EsaProcessor.RETURN_NUM` candidate Wikipedia concepts

#### `ElmoProcessor`

Supports all operations related to ElMo and its data files.

A main entrance is `ElmoProcessor.rank_candidates`, which given a sentence and a list 
of candidates (generated from ESA), rank them by ELMo representation cosine similarities. (see paper)

It will return the top `ElmoProcessor.RANKED_RETURN_NUM` candidates.

#### `InferenceProcessor`

This is the core engine that does inference given outputs from the previous processors.

The logic behind it is as described in the paper and is rather complicated. 

One main entrance is `InferenceProcessor.inference` which receives a sentence, outputs from 
previously mentioned processors, and set inference results.

#### `Evaluator`

This evaluates performances and print them, after given a list of sentences processed by
`InferenceProcessor`

#### `DataReader`

Initialize this with a data file path. It reads standard json formats (see examples)
and transform the data into a list of `Sentence`

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
