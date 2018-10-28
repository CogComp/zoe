#!/bin/bash

if ! [ -x "$(command -v java)" ]; then
    echo 'Error: Java in not installed.'
    exit 1
fi
if ! [ -x "$(command -v mvn)" ]; then
    echo 'Error: maven is not installed.'
    exit 1
fi
if ! [ -x "$(command -v python3)" ]; then
    echo 'Error: python 3.x is not installed.'
    exit 1
fi
if ! [ -x "$(command -v virtualenv)" ]; then
    echo 'Error: virtualenv is not installed.'
    exit 1
fi
if ! [ -x "$(command -v wget)" ]; then
    echo 'Error: wget is not found. Either install or find replacement and modify this script.'
    exit 1
fi
if ! [ -x "$(command -v unzip)" ]; then
    echo 'Error: unzip is not found. Either install or find replacement and modify this script.'
    exit 1
fi
echo 'All dependencies satisfied. Moving on...'

virtualenv -p python3 venv
cd ./bilm-tf
../venv/bin/python3 setup.py install
wget http://cogcomp.org/Data/ccgPapersData/xzhou45/zoe/model.zip
unzip model.zip
rm model.zip
cd ../
venv/bin/pip3 install Cython
venv/bin/pip3 install -r requirements.txt
wget http://cogcomp.org/Data/ccgPapersData/xzhou45/zoe/data.zip
unzip -n data.zip
rm data.zip
python -m ccg_nlpy download