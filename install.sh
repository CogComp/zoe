#!/bin/bash

virtualenv -p python3 venv
cd ./bilm-tf
../venv/bin/python3 setup.py install
wget http://cogcomp.org/Data/ccgPapersData/xzhou45/zoe/model.zip
unzip model.zip
rm model.zip
cd ../
venv/bin/pip3 install -r requirements.txt
wget http://cogcomp.org/Data/ccgPapersData/xzhou45/zoe/data.zip
unzip -n data.zip
rm data.zip