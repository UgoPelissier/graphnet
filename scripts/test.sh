#!/usr/bin/env bash

wdir=/home/eleve05/safran/graphnet
ckpt_path=/home/eleve05/safran/graphnet/logs/version_7/checkpoints/epoch=900-step=169388.ckpt

cd $wdir
clear
python main.py test --ckpt_path $ckpt_path