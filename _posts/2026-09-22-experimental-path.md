---
layout: "post"
title: "Move Fast. Don\u2019t Break Things."
date: "2026-09-22 00:00:00 +0000"
body_class: "technical-blog"
toc: true
excerpt: "Introducing FlashInfer's Experimental Path"
author: "Brian K. Ryu, Alex Yang, Anerudhan Gopal, Yang Xu, Jingfan Sun"
---

*Introducing FlashInfer's Experimental Path*

New kernels should reach users faster, while stable workloads should keep working as expected. FlashInfer’s experimental path is designed to deliver both. New architectures and GPU generations create new operator variants and optimization opportunities, while systems such as CAKE and agent-generated libraries such as b12x accelerate the supply of candidate kernels. The bottleneck is increasingly not writing a kernel, but evaluating, integrating, and delivering it safely.

Moving too aggressively risks regressions and maintenance burden, while requiring every new kernel to satisfy the full stable-release standard upfront slows support for new models, operators, and GPU architectures. With only one release path, specialized kernels must either wait for every stable requirement or enter through one-off exceptions—neither of which scales with the growing volume of human-authored, agent-assisted, and generated kernels.

FlashInfer v0.7 introduces an experimental path to bridge this gap. It gives contributors a defined route for landing new functionality sooner while keeping experimental code opt-in, isolated, testable, and accountable. The goal is not to lower the bar for stable FlashInfer APIs. Instead, the experimental path creates a deliberate incubation path between a promising kernel and a broadly supported library feature.

## Introducing the design of FlashInfer.experimental

The design starts from three considerations. First, API stability and backend maturity are separate: a stable API can expose a new experimental implementation without changing its public contract. Second, experimental behavior must be an explicit opt-in, so existing users never enter a less mature path silently. Third, experimental code must remain contained so that it can graduate or be removed without becoming entangled with stable functionality.

This produces a small but important matrix. A stable API with a stable backend follows the normal path. A stable API may also expose an experimental backend with explicit opt-in. A new experimental API is paired with an experimental implementation.

| **API / backend status** | **Stable backend** | **Experimental backend** |
| --- | --- | --- |
| **Stable API** | Normal path | Allowed with explicit opt-in |
| **Experimental API** | Not a target use case | Allowed with explicit opt-in |

Experimental API entry points can now be added by marking the new API with `@flashinfer_experimental_api`, so graduation from experimental to stable-core does not require an import-path change. Experimental backends and backend-specific routing, compilation, caching, and kernels live under `flashinfer.experimental`. Core entry points remain thin and defer importing experimental implementations. The full rationale and design are documented in the [FlashInfer experimental API and backend design document](https://github.com/flashinfer-ai/flashinfer/blob/main/docs/design_docs/experimental_apis_and_backends.md).

### Marking an API or backend as experimental

The following illustrative snippets show the registration pattern; helper functions and feature modules are application-specific. A new public operation uses `@flashinfer_experimental_api` and delegates to an implementation under `flashinfer.experimental`:

```python
## Existing stable-core API
from flashinfer.api_logging import flashinfer_api

@flashinfer_api # Stable API
def existing_op(x, **kwargs):
    return run_existing_op(x, **kwargs)

## [NEW] Mark API as experimental
from flashinfer.api_logging import flashinfer_experimental_api

@flashinfer_experimental_api
def my_new_op(x, **kwargs):
    # API is a thin entry point.
    from .experimental.my_feature import run
    return run(x, **kwargs)
```

An implementation of an existing stable API uses `@experimental_backend` on its lightweight support checker:

```python
from flashinfer.experimental import experimental_backend
from flashinfer.utils import supported_compute_capability

import torch

# In the lightweight support module:

@experimental_backend # [NEW] Mark backend as experimental
@supported_compute_capability([120, 121])
def check_sm12x_cute(a, b, out=None, backend="auto"):
    return a.dtype == torch.bfloat16 and a.shape[-1] % 64 == 0

@backend_requirement(
    backend_checks={"cutlass": _check_cutlass, "sm12x_cute": check_sm12x_cute},
    heuristic_func=_heuristic_mm_bf16,
)
@flashinfer_api # Existing stable API
def mm_bf16(a, b, out=None, backend="auto"):
    if backend == "auto": # Existing stable backend
        backend = mm_bf16.suitable_auto_backends[0]
    if backend == "sm12x_cute": # New experimental backend
        from ..experimental.sm12x_gemm import run  # deferred import
        return run(a, b, out)
    ...
```

### Opt in explicitly; keep stable behavior stable

Calling an experimental API or explicitly naming an experimental backend is itself an opt-in. FlashInfer emits an `ExperimentalWarning` on first use. Importing `flashinfer.experimental` remains safe for tooling and introspection; the warning occurs when experimental functionality is used.

Automatic backend selection is more conservative. When a stable API uses automatic backend selection (`backend="auto"`) or does not have an explicit backend selection, FlashInfer does not consider experimental backends or kernels unless the user sets `FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1`. The same boundary applies to dispatch and autotuning, preventing an experimental backend from unexpectedly becoming the selected implementation for an existing stable workload.

This lets developers evaluate a promising backend with a targeted code change and return to the stable path by removing that selection. Supported framework releases can keep experimental functionality disabled by default until it is ready to graduate.

### Two CI lanes preserve one stable contract

Every experimental feature must pass reference-based correctness tests within its declared support scope. Admission also requires at least one representative configuration, validation on the intended hardware, and a runnable example. Experimental implementations use a focused CI lane, while stable-core changes retain the existing review, compatibility, testing, and support requirements.

| **Stable lane** | **Experimental lane** |
| --- | --- |
| **Stable API or backend change**<br>Long-term support commitment | **Experimental PR + tracking issue**<br>Owner, narrow scope, and graduation plan |
| **Full review and stable CI**<br>Broad compatibility, documentation, trace, and packaging checks. | **Focused review + experimental CI**<br>Reference correctness, target hardware, and runnable example |
| **Stable release path**<br>Normal routing, long-term support, and AOT eligibility | **Explicit opt-in, JIT-only**<br>Stable CI remains isolated; graduation moves the feature into the stable lane |

Experimental tests live under `tests/experimental/` and do not participate in stable test discovery. Stable support, performance validation, and the complete documentation surface become graduation requirements rather than admission requirements.

### The incubation/graduation process

Experimental does not mean permanent preview. Every experimental feature requires a named owner and a tracking issue that documents its use case, the reason for entering the experimental path, and a graduation plan. The default intent is for a feature to graduate within four weeks.

At each lifecycle review, the feature can graduate through the normal stable process, continue incubating with maintainer approval, or be removed. An extension requires evidence of active use or progress and a clear list of remaining blockers. Broken tests, loss of ownership, or lack of meaningful usage are valid reasons for removal.

Graduation restores the full stable contract: the API is finalized, backend code moves to its stable location, trace and documentation requirements are completed, tests join the stable suite, and the feature is registered for AOT packaging when appropriate.

## How developers can use the experimental path

The infrastructure and policy introduced in [PR #4880](https://github.com/flashinfer-ai/flashinfer/pull/4880) provide the common framework for these contributions. The [normative policy](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/experimental/README.md) describes admission, opt-in behavior, testing, ownership, and graduation.

For users, begin with the documentation for the specific feature. Confirm whether the API, backend, or both are experimental; check the supported GPU, precision, shapes, dependencies, and routing behavior; and start with the provided example. Select an experimental backend explicitly for initial evaluation. Enable automatic experimental backend selection only when that broader opt-in matches the deployment.

For contributors, start with a tracking issue and a narrow support claim. Include a reference implementation, correctness tests, one representative validated configuration, intended-hardware evidence, and a runnable example. Keep backend-specific logic inside flashinfer.experimental and stable-core changes minimal. Mark the pull request for the experimental track and link its tracking issue.

### What is entering through the fast path

The following contributions include ongoing work beyond the v0.7 release. They illustrate the kinds of narrow or fast-moving contributions this path is designed to admit:

- FP16 and FP8 attention for SM110 on NVIDIA Jetson AGX Thor ([PR #5302](https://github.com/flashinfer-ai/flashinfer/pull/5302), [PR #5293](https://github.com/flashinfer-ai/flashinfer/pull/5293)) — architecture-specific kernels with narrow support surface.
- NVFP4 attention for SM103 ([PR #5283](https://github.com/flashinfer-ai/flashinfer/pull/5283)) — architecture-specific kernels with narrow support surface.
- MXFP8 MegaMoE EP16 for SM103 ([PR #4970](https://github.com/flashinfer-ai/flashinfer/pull/4970)) — a specialized MegaMoE configuration.
- CUTLASS PrimsTS attention for SM100 ([PR #5082](https://github.com/flashinfer-ai/flashinfer/pull/5082)) — a new implementation stack built on recently introduced CUTLASS DSL APIs.
- Unified KDA Wrapper `RecurrentKDAPrefillWrapper` API ([PR #5040](https://github.com/flashinfer-ai/flashinfer/pull/5040)) — a new public API whose contract is still being validated.

Experimental does not mean a different correctness standard. Human-authored, agent-assisted, and generated kernels follow the same contract: explicit scope, correctness evidence, ownership, isolation, and a path toward graduation.

## From faster generation to faster adoption

Coding agents and kernel-generation systems can shorten the time required to create and optimize GPU kernels. The larger opportunity is to shorten the time between a credible implementation and useful feedback from real inference workloads.

`flashinfer.experimental` turns faster kernel generation into faster, controlled adoption. It protects stable behavior while giving credible experiments an explicit support envelope, an owner, and a path to graduation.

Explore the experimental API and backend policy, review the active experimental pull requests, and try a feature whose documented support envelope matches your workload. Share correctness and performance results through the feature's tracking issue so the community can decide what should graduate next.

## Acknowledgements

We thank the FlashInfer team for their contributions to the design of FlashInfer's Experimental Path.

- Brian K. Ryu: design and implementation of the experimental API and backend path.
- Alex Yang: design review and implementation of the experimental path.
- Anerudhan Gopal: design review and API contract discussions.
- Yang Xu and Jingfan Sun: design feedback and engineering leadership.

## References

- Experimental APIs and backends infrastructure and policy: [PR #4880](https://github.com/flashinfer-ai/flashinfer/pull/4880)
- [Experimental API and backend design document](https://github.com/flashinfer-ai/flashinfer/blob/main/docs/design_docs/experimental_apis_and_backends.md)
- [Normative FlashInfer experimental policy](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/experimental/README.md)


---

Part of [FlashInfer v0.7](/2026/09/22/flashinfer-v07.html). See the [release highlights](/releases/#v0-7-0) for upgrade details.
