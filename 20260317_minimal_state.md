# 2026-03-17 FreiHAND complete run: minimal code state

## Target run

Target log:
- `/root/code/vepfs/GPGFormer/logs/ablations/freihand_loss_beta_mesh_target51_recommended_sidetuning_20260317.log`

Important note:
- This log contains two launches.
- The incomplete first launch starts at `0/542`.
- The complete launch starts at `0/508`.
- The complete launch reaches the final best result at epoch 59:
  - `PA-MPJPE=5.121`
  - `PA-MPVPE=5.586`
  - gap=`0.465`

Evidence from the log:
- `0/542` first launch: line 72
- `0/508` complete launch: line 247
- `zero-init enabled`: lines 58-61 and 233-236
- optimizer signature for the complete launch: line 243
- best checkpoint at epoch 59: line 62767


## Main conclusion

The 2026-03-17 complete run does **not** match any clean git commit currently in history.

The closest clean base is:
- `d9aa6d3cc209fbf9dbd57c26c7129cc4d801a9d6`

But the actual 2026-03-17 run was a **dirty local tree** built on top of that era, with a very specific combination:

1. It already had:
   - `sum_geo_gate`
   - side-branch LR ramp logging
   - deterministic training enabled from config
   - `geo_side_adapter` support

2. It did **not yet** have:
   - `small-init` in `wilor_vit.py`
   - the separate `gate` optimizer group
   - the simplified `[info]` print that removed `ho3d_use_json_split`


## Why this fingerprint is reliable

The 2026-03-17 complete run shows all of the following at the same time:

1. `zero-init enabled`
   - So it is earlier than the later `small-init` change.

2. `geo_fusion lr=... (16 tensors)` and **no** `gate lr=...`
   - So `encoder.sum_geo_gate` was still counted inside `geo_fusion`, not split into its own optimizer group.

3. `side lr=... (6 tensors)`
   - This matches the current `GeoSideAdapter` config with `depth: 2`.
   - It does **not** match the old committed `GeoSideTuning` path cleanly.

4. `side_tuning_start_factor=0.100`
   - So the code already had the later side-branch ramp scheduler logic.

5. `ho3d_use_json_split=False` printed in the `[info]` line
   - So it was still using the older train.py info-print path, not the current simplified one.

This combination uniquely points to a mixed local state, not a clean historical commit.


## Minimal state relative to the closest clean base

Closest clean base:
- commit `d9aa6d3`

To recover the 2026-03-17 complete run state from that base, the minimum required local changes are:

1. Add deterministic seeding support.
   - Files:
     - `gpgformer/utils/distributed.py`
     - `train.py`
   - Needed because the log shows deterministic warnings during training.

2. Add `geo_side_adapter` support and enable it from config.
   - Files:
     - `gpgformer/models/tokenizers/geo_side_adapter.py`
     - `gpgformer/models/gpgformer.py`
     - `train.py`
   - Needed because the log shows `side lr=... (6 tensors)`, which matches `GeoSideAdapter(depth=2)`.

3. Keep `sum_geo_gate` inside the `geo_fusion` optimizer group.
   - File:
     - `train.py`
   - Needed because the log shows `geo_fusion ... (16 tensors)` and no separate `gate` group.

4. Keep fusion projection as strict zero-init.
   - File:
     - `gpgformer/models/encoders/wilor_vit.py`
   - Needed because the log prints `zero-init enabled`, not `small-init enabled`.


## Minimal state relative to the current worktree

If the goal is to reconstruct the 2026-03-17 complete run from the **current** worktree, the minimum changes are:

### Keep as-is

These parts of the current worktree are consistent with the 2026-03-17 run and should stay:

1. `gpgformer/models/tokenizers/geo_side_adapter.py`
2. `gpgformer/models/gpgformer.py`
   - keep `GeoSideAdapter` import/config/wiring
3. `train.py`
   - keep `geo_side_adapter` config parsing and model wiring
   - keep `geo_side_adapter` parameters counted as `side` parameters
4. `gpgformer/utils/distributed.py`
   - keep deterministic seeding support
5. config path:
   - `configs/config_freihand_loss_beta_mesh_target51_recommended_sidetuning.yaml`

### Revert

These current changes are later than the 2026-03-17 complete run and should be reverted for a faithful reconstruction:

1. `gpgformer/models/encoders/wilor_vit.py`
   - revert the last fusion layer init from:
     - `nn.init.normal_(..., std=1e-3)`
   - back to:
     - `nn.init.zeros_(...)`
   - revert log text from:
     - `small-init enabled`
   - back to:
     - `zero-init enabled`

2. `train.py`
   - remove the separate `gate` optimizer group
   - put `encoder.sum_geo_gate` back into `geo_fusion`
   - remove `gate lr=...` logging

3. `train.py`
   - restore the older info print:
     - `[info] config=... dataset.name=... ho3d_use_json_split=...`
   - The current simplified print does not match the 2026-03-17 log signature.


## Config fingerprint for the 2026-03-17 complete run

The config path is the same one used today:
- `configs/config_freihand_loss_beta_mesh_target51_recommended_sidetuning.yaml`

The current file already matches several key 2026-03-17 fingerprints:

1. `fusion_proj_zero_init: true`
2. `sum_geo_gate_init: -1.2`
3. `side_tuning.enabled: false`
4. `geo_side_adapter.enabled: true`
5. `geo_side_adapter.depth: 2`
6. `side_tuning_ramp_epochs: 12`
7. `deterministic: true`
8. `out_dir: ..._20260317`

Why this matters:
- `geo_side_adapter.depth: 2` implies 6 parameter tensors for the side branch.
- That matches the 2026-03-17 log line:
  - `side lr=... (6 tensors)`


## File-level minimal set

For a faithful 2026-03-17 reconstruction, the minimum file set is:

1. `train.py`
2. `gpgformer/models/encoders/wilor_vit.py`
3. `gpgformer/models/gpgformer.py`
4. `gpgformer/models/tokenizers/geo_side_adapter.py`
5. `gpgformer/utils/distributed.py`
6. `configs/config_freihand_loss_beta_mesh_target51_recommended_sidetuning.yaml`


## Short reconstruction recipe

The shortest accurate description is:

`2026-03-17 complete run = current sidetuning config + geo_side_adapter-enabled local code + deterministic local code + zero-init + no gate split`

In other words:

- Base behavior was already beyond clean `d9aa6d3`.
- But it was still earlier than the later `small-init` and `gate lr` changes now visible in the current worktree.


## Not required for the minimal March 17 fingerprint

The following current worktree changes are not needed to explain the 2026-03-17 result:

1. worker seeding helper `_seed_worker`
2. dataloader `generator=loader_generator`
3. the later HO3D loader simplification
4. the later simplified `[info]` log format

