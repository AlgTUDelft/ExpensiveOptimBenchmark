#!/bin/bash
source /opt/openfoam4/etc/bashrc
res=$(python3 dockerCall.py -p $1 $2 2>&1)
#python3 dockerCall.py -p $1 $2

if [[ $res == *"dockerCall.py"* ]]; then
  echo "Error"
else
  #echo $res
  if [[ $1 == "PitzDaily" ]]; then
    echo $res | grep -Eo '[+-]?[0-9]+([.][0-9]+)?(e-[0-9]+)?' | head -n 1
  fi
  if [[ $1 == "KaplanDuct" ]]; then
    echo $res | grep -Eo '[+-]?[0-9]+([.][0-9]+)?(e-[0-9]+)?' | tail -n 1
  fi
  if [[ $1 == "HeatExchanger" ]]; then
    echo $res | grep -Eo '[+-]?[0-9]+([.][0-9]+)?(e-[0-9]+)?' | head -n 4 | tail -n 2
  fi
  if [[ $1 == "ESP" ]]; then
    echo $res | grep -Eo '[+-]?[0-9]+([.][0-9]+)?(e-[0-9]+)?'
  fi
  if [[ $1 == "ESP2" ]]; then
    echo $res | grep -Eo '[+-]?[0-9]+([.][0-9]+)?(e-[0-9]+)?'
  fi
fi
