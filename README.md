# ZOE
A state of the art system for zero-shot entity fine typing with minimum supervision

## Introduction

This is a demo system for our paper "Zero-Shot Open Entity Typing as Type-Compatible Grounding",
which at the time represents the state-of-the-art of zero-shot entity typing.

The original experiments are done is a package written in Java. This is a re-written package 
that contains the same core, without code for experiments or cache generations etc, solely for
the purpose of demoing the algorithm and validating the results. 

The results may slightly differ from published numbers, due to the randomness in Java's 
Hashset iteration order. The difference should be within 0.5%.

A major flaw of this demo is the speed. It's much slower than what people normally
expect for a production system, due to the usage of naive data structures like Python lists.
Re-written with modern packages written in Cython like numpy may hugely improve it.

## Usage

### Download the required data files

UNDER CONSTRUCTION

### Install the system

You need to first install ELMo packages by running `python3 setup.py install` in bilm-tf directory

Then install requirements by `pip3 install -r requirements.txt` in project root

Then check if all files are here by `python3 scripts.py CHECKFILES` or `python3 scripts.py CHECKFILES figer`
in order to check figer caches etc.

### Run the system

The system supports generating results in multiple settings. 

Currently you can do the following:
* Run FIGER experiments: `python3 main.py figer` (note this usually takes 2 hours)

Please refer to `main.py` to see how you can test on your own data.

## Engineering details

### Structure

The package is composed with 

* A slightly modified ELMo source code, (see bilm-tf/)
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

A main eentrace is `ElmoProcessor.rank_candidates`, which given a sentence and a list 
of candidates (generated from ESA), rank them by ELMo representation cosine similarities.
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

## Reference 

UNDER CONSTRUCTION
