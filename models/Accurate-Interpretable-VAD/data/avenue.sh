#!/bin/bash

echo "Downloading CUHK-Avenue dataset....."

cd ./data

wget "http://101.32.75.151:8181/dataset/avenue.tar.gz"
tar -xvf avenue.tar.gz
rm avenue.tar.gz
rm avenue/avenue.mat
cp avenue.mat avenue/

echo "Download CUHK-Avenue successfully!"