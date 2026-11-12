#!/bin/bash
# O DNS desta máquina falha de vez em quando: repete até baixar tudo (o hf retoma de onde parou).
export HF_HUB_ENABLE_HF_TRANSFER=1
for i in $(seq 1 30); do hf download Agent-Ark/Toucan-1.5M --repo-type dataset --local-dir /workspace/dados/toucan && break; sleep 10; done
for i in $(seq 1 30); do hf download SWE-bench/SWE-smith-trajectories --repo-type dataset --local-dir /workspace/dados/swe-smith-traj && break; sleep 10; done
ls /workspace/dados/toucan/*/*.parquet | wc -l > /workspace/dados/BAIXADO
