#!/usr/bin/env bash
BASEDIR=$(dirname "$0")

sudo apt-get install python3-venv
python3 -m venv venv
source $BASEDIR/venv/bin/activate
python3 -m pip install --upgrade pip
pip install Keras==2.4.3
pip install matplotlib==3.3.4
pip install plotly
pip install PyQt5
pip install tensorflow
pip install pandas
python3 jb.py
