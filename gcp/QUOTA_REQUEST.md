# GPU Quota Request — NanoLLM

## Current state

| Quota | Scope | Limit | Status |
|---|---|---|---|
| `GPUS_ALL_REGIONS` | global | 0 | **Denied** (first request, no justification given) |
| `NVIDIA_L4_GPUS` | us-central1 | 1 | Not yet requested |
| `CustomModelServingL4GPUs` | us-central1 | 4 | Approved — but this is Vertex AI *serving*, not training |

`GPUS_ALL_REGIONS = 0` blocks **every** GPU VM in the project, including a
single-GPU one. It has to be raised before anything else matters.

`NVIDIA_L4_GPUS` defaults to 1 in every region, so it must be raised
separately — a granted global quota alone still caps you at one GPU per region.

## What to request

Console → IAM & Admin → Quotas → filter by metric name.
<https://console.cloud.google.com/iam-admin/quotas?project=nanollm-507220>

**Request 1 — global**
- Service: `Compute Engine API`
- Metric: `GPUS_ALL_REGIONS` (shown as "GPUs (all regions)")
- Dimension: none (global)
- New limit: **4**

**Request 2 — regional**
- Service: `Compute Engine API`
- Metric: `NVIDIA_L4_GPUS` (shown as "NVIDIA L4 GPUs")
- Dimension: `region = us-central1`
- New limit: **4**

Both must be granted. A grant on one alone changes nothing.

## Justification (paste this)

> Training a small GPT-2 style language model (~30M parameters) on the
> TinyStories dataset as an open-source machine learning engineering project.
>
> I need 4 NVIDIA L4 GPUs in us-central1 on a single g2-standard-48 instance to
> run distributed data-parallel (DDP) training with PyTorch, and to measure
> throughput scaling across 1, 2 and 4 GPUs. The project also includes custom
> CUDA kernels for LayerNorm which must be validated and profiled with Nsight
> Compute on real Ada-generation hardware (sm_89).
>
> L4 specifically is required because the training loop runs in bfloat16.
> NVIDIA T4 is Turing (sm_75) and has no hardware bfloat16 support, so it
> cannot run this workload correctly. L4 is the lowest-cost GCP GPU with native
> bfloat16.
>
> Expected usage is short, bounded runs — a few hours per training job, with
> the instance stopped between runs. Billing is enabled and in good standing.
> Estimated spend is under $100/month.

## Tips that improve approval odds

- **Never leave the justification blank.** The first request was denied with an
  empty justification field; that alone is a common cause.
- Name the exact machine type (`g2-standard-48`) and region. Vague requests
  read as speculative.
- State *why this specific GPU model* — the bf16/sm_75 argument above is a
  concrete technical constraint, not a preference.
- Ask for a modest number. 4 is routinely granted; 8+ draws more scrutiny.
- Make sure billing is enabled first (it is) — requests on projects without
  billing are auto-denied.
- If it is denied again, request `NVIDIA_L4_GPUS = 2` and
  `GPUS_ALL_REGIONS = 2` instead. Smaller asks clear more easily, and 2 GPUs is
  still enough to demonstrate DDP.

## Fallback if Compute Engine is denied again

Vertex AI runs on a **separate quota pipeline**, and it already approved 4x L4
for serving on this project — so it is demonstrably friendlier here. The metric
for training jobs is:

- Service: `Vertex AI API`
- Metric: `CustomModelTrainingL4GPUsPerProjectPerRegion`
- Dimension: `region = us-central1`
- New limit: **4**

Vertex custom training supports multi-GPU workers, so DDP works there too — it
needs the trainer packaged as a container rather than run over SSH.

## Verify

```bash
./gcp/check_quota.sh
```

Prints effective limits and the status of every request you have filed.
