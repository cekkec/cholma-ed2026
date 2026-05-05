#!/bin/bash
# Usage: sbatch run_train.sh <strategy>
#   strategy ∈ {CA, MA, Soft}
#     CA    : train Complete-Agreement, val Complete-Agreement
#     MA    : train Majority-Agreement, val Majority-Agreement
#     Soft  : train SoftLabel,          val SoftLabel

python main.py
