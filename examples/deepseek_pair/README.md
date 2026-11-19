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

### Reusing generated answers and edited conversations

Set `DEEPSEEK_PREFIX_DECODE_END=1` on both ranks to snapshot completed or
cancelled requests before releasing their state. The snapshot covers only
tokens already evaluated by the model: the last sampled token is evaluated
again on the next request. This lets an agent reuse its previous answer in
addition to its input prompt. It does not retain cancelled work that has not
been evaluated, or work lost when the server process exits.

`DEEPSEEK_PREFIX_STRIDE=1024` additionally saves intermediate checkpoints at
scheduler boundaries after at least 1024 new computed tokens. An edited or
branched conversation can restore the longest stored exact prefix shared by
both ranks, then evaluate its remaining suffix. This is checkpoint-based
reuse, not arbitrary rewind of a later snapshot; scheduler chunk sizes affect
checkpoint spacing. Set the stride to zero (the default) for completion-only
snapshots. Both options require `DEEPSEEK_PREFIX_CACHE=1`.

The existing RAM and SSD budgets bound these additional snapshots. SSD writes
remain synchronous, so measure their cost against follow-up TTFT and avoid
overly frequent checkpoints. Checkpoints share the existing LRU with prompts.
Both ranks negotiate the same feature configuration at startup (protocol 5).

Run snapshot, disk, branch-isolation and continuation tests against the matching
MLX VLM source checkout with:

```sh
python -m pytest -q examples/deepseek_pair/test_prefix_cache.py
```

## Distributed greedy sampling

Set `DEEPSEEK_SHARDED_GREEDY=1` on both ranks to reduce vocabulary traffic
for plain greedy requests. Each rank projects its original FP32 vocabulary
shard and exchanges its best candidate. When the client requests up to five
top logprobs the ranks also exchange normalization statistics and top candidates.
The option covers ordinary decode and DSpark target verification. A batch with
random sampling, grammar constraints, penalties, or unsupported logprob options
uses the existing full-vocabulary sampler. The default is off. The service logs
the candidate and full-logit byte counts once per shape for comparison.

## DSpark greedy serving

Set `DEEPSEEK_DSPARK_TOKENS=5` on both ranks and add
`--speculative-config '{"method":"dspark","num_speculative_tokens":5}'` to
the leader. The values must match. Zero (the default) disables drafting.
Use a matching MLX VLM revision that supports block transactions on
`BatchDeepseekV41Cache`; both ranks check source signatures before serving.

The worker returns proposals to the vLLM scheduler, verifies scheduled blocks
with the target model, and commits each request's accepted prefix independently.
It returns accepted tokens plus the target's correction/bonus token, including
matching logprobs and stop/budget handling. `dspark_verify` events and vLLM's
speculative metrics report actual verification, rather than just a loaded draft
model. Requests with nonzero temperature, structured output, or repetition
detection retain ordinary decoding. Stochastic rejection sampling is not
implemented by this adapter.

Target prefix snapshots remain enabled. Draft caches are request-local and
discarded when a request finishes; restored target snapshots do not include
draft hidden states. The drafter uses the newly evaluated suffix in that case,
so measure acceptance on cache-hit workloads separately. All emitted tokens
still pass the target verifier. Target verification groups requests with equal
scheduled block widths.

With a positive draft width, two additional options are available on both ranks:

- `DEEPSEEK_DSPARK_BATCH=1` groups equal-width proposals into one drafter
  forward pass, with independent request positions and rotating KV caches.
- `DEEPSEEK_DSPARK_ADAPTIVE=1` explores ordinary decode, up to two draft
  tokens, and the configured maximum. It tracks elapsed time per emitted token
  for each active batch size and periodically samples the other widths.
  Inspect `dspark_policy_sample` events for width, elapsed time and emitted
  tokens. Paused drafts retain the latest target hidden window for resumption.

Both options default to off and can be enabled independently.

Run the worker's fixed-logit acceptance, stop, logprobs, and mixed-sampling
regressions in the serving Python environment without loading model weights:

```sh
TORCH_COMPILE_DISABLE=1 PYTHONPATH=examples/deepseek_pair \
  python -m unittest discover -s examples/deepseek_pair -p test_worker.py
```
