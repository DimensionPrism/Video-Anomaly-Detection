#!/bin/bash

echo "Downloading UCSD-Ped2 dataset....."

cd ./data

wget "http://101.32.75.151:8181/dataset/ped2.tar.gz"
tar -xvf ped2.tar.gz
rm ped2.tar.gz

echo "Download UCSD-Ped2 successfully!"