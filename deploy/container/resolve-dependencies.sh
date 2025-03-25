#! /bin/bash

apt update
apt install -y libgl1-mesa-dev libglib2.0-0 protobuf-compiler libprotobuf-dev cmake
pip3 install torch
pip3 install yolox