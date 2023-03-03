#!/bin/bash

python -m pip install -r requirements.txt
python setup.py
conda install -y -c pytorch faiss-gpu