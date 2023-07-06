#!/usr/bin/env bash

wdir=/home/eleve05/safran/graphnet
ckpt_path=/home/eleve05/safran/graphnet/logs/version_6/checkpoints/epoch=999-step=188000.ckpt

cd $wdir
clear
python main.py test --ckpt_path $ckpt_path