# Experimental two-Mac DeepSeek V4.1 worker

This adapter connects the vLLM V1 scheduler and sampler to two MLX JACCL
ranks. It is an experimental deployment integration, not the standard Metal
paged-attention backend. The scheduler's KV blocks account for requests;
actual compressed KV, Engram history, and cross-layer handoffs belong to MLX.

Requires the DeepSeek V4.1 implementation from `des-intelli/mlx-vlm`, initially
based on commit `6f93965f97d31b97ab25c5b512873011feece8de` (upstream PR 2406).
The tested environment is MLX 0.32.4, MLX LM 0.32.3.dev12+g4cfc64784,
vLLM 0.33.0+cpu and vLLM Metal 0.33.0.dev20261118060554. Keep these versions
fixed while comparing optimizations; installing this fork's current main is
not required to use the adapter.

Set `DEEPSEEK_MODEL`, `DEEPSEEK_CLUSTER_ROOT`, `DEEPSEEK_LEADER`,
`DEEPSEEK_PEER`, `MLX_RANK`, `MLX_JACCL_COORDINATOR`, and `MLX_IBV_DEVICES`.
The root directory contains a shared `pair.auth` secret with restricted file
permissions. Control traffic must stay on a trusted private network. Start
`python -u shard.py` on rank 1 and use `--worker-cls studio_worker.StudioWorker`
with `--distributed-executor-backend uni` on rank 0. Add this directory to
`PYTHONPATH` on both ranks. Both ranks must run identical source versions.

Keep `--no-enable-prefix-caching --no-async-scheduling --enforce-eager`:
standard vLLM shared cache blocks and async model execution are not implemented
by this adapter. This example supports generation only.

Distribute pinned source artifacts over existing SSH. Public read-only GitHub
clones need no key; never place developer write credentials on serving nodes.
