#!/usr/bin/env bash

wdir=/home/upelissier/30-Code/graphnet/
ckpt_path=/data/users/upelissier/30-Code/graphnet/logs/version_0/checkpoints/epoch=900-step=169388.ckpt

cd $wdir
clear
python main.py test --ckpt_path $ckpt_path