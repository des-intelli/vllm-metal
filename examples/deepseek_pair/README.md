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

## Request-prefix snapshots and SSD tier

Set `DEEPSEEK_PREFIX_CACHE=1` on **both** ranks to enable the adapter's own
prefix store. It snapshots compressed KV buffers, sliding windows, partial
compression state, Engram history, and cross-layer handoffs immediately before
the final prompt token. Snapshots use exact token-ID hashes. Only prefixes
successfully restored on both ranks are reused; missing or unreadable entries
fall back to normal prefill. The last prompt token is always evaluated to obtain
fresh logits. Separate request copies protect snapshots from later mutation.

Configure each rank with:

- `DEEPSEEK_PREFIX_MEMORY_BYTES`: RAM LRU budget (default 2 GiB).
- `DEEPSEEK_PREFIX_DISK_BYTES`: SSD budget (default zero, disabled).
- `DEEPSEEK_PREFIX_DIR`: service-owned cache directory.
- `DEEPSEEK_PREFIX_NOCACHE=1`: request Darwin `F_NOCACHE` for disk reads during
  I/O experiments. This is not a global filesystem-cache purge or proof of a
  physically cold SSD read.

Budgets apply separately per rank and model/source identity. Old identities
remain in separate directories for explicit cleanup. Do not run multiple
writers against one rank's cache directory. Checkpoint files must be immutable
while serving; identity includes configuration/tokenizer bytes, weight-file
sizes and modification times, model source and MLX version. Files contain tensor
data in safetensors format and are created with owner-only permissions. Writes
are synchronous and atomic. `prefix_save` records their bytes and latency;
`prefix_restore` records the tier and read time. `prefix_hit` records tokens
actually reused after both-rank confirmation.

Keep the vLLM `--no-enable-prefix-caching` flag: scheduler blocks are still
virtual accounting objects. Cached prompt positions advance through normal
scheduler steps without model execution. Consequently standard vLLM cached-token
metrics do not describe this store; use the adapter's prefix events. Prefix
caching is opt-in, intended for this private experimental service. It does not
implement a general distributed vLLM cache connector.

Run snapshot, disk, branch-isolation and continuation tests against the matching
MLX VLM source checkout with:

```sh
python -m pytest -q examples/deepseek_pair/test_prefix_cache.py
```
