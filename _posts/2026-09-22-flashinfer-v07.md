---
layout: post
title: "Accelerate LLM Inference with Open Kernels and Smarter Autotuning with FlashInfer v0.7"
date: 2026-09-22 01:00:00 +0000
author: "Alex Yang, Brian K. Ryu, Anerudhan Gopal, Kaustubh Rao, Yang Xu, Jingfan Sun, Albert Cheng, Vincent Tombari, Md Saidul Hoque Anik, Lee Nau, Xin Li, Po-Han Huang"
body_class: technical-blog
toc: true
excerpt: "Open kernels, an experimental contribution path, unified expert-parallel MoE, and serving-aware autotuning in FlashInfer v0.7."
---

FlashInfer’s mission is to deliver high-performance GPU kernels across inference stacks, keeping pace with the frontier of model development. For inference-engine developers, the motivation is straightforward: bring new models into service sooner and keep improving inference performance as models and hardware evolve.

Meeting that goal is increasingly complex. The best kernel configuration depends on the model architecture, precision, GPU generation, and serving mode—and each continues to change. Developers need access to kernels for newly released models, a reliable way to select configurations for their serving setup, and a consistent API for dispatching tokens to experts and combining their outputs across MoE implementations.

FlashInfer v0.7 addresses these challenges with four connected features:

- An **experimental path** that makes new kernels, including agent-assisted contributions, available sooner as explicit opt-ins with a documented graduation policy.
- **Open attention and MoE kernels** built on CUTLASS Primitives and Task Scheduling that developers can inspect, benchmark, and extend.
- A **unified MoE expert-parallel API**, including fused MegaMoE execution that overlaps communication with compute.
- **Autotuner v2**, which lets users choose a measurement policy matching eager execution or CUDA Graph replay for more accurate kernel ranking, and reuses compatible tuning results across restarts.

This article introduces those features and the accompanying technical posts. For the complete upgrade summary, including architecture-specific packaging, compatibility changes, and removals, see the [v0.7.0 release highlights](/releases/#v0-7-0).

## Try new kernels sooner with an experimental path

New models and GPU generations create new operator variants and optimization opportunities. Agent-assisted kernel authoring adds to that supply, but promising implementations still need correctness checks, hardware validation, integration, and feedback from real workloads.

FlashInfer’s experimental path gives contributors a defined route for landing functionality sooner while keeping it opt-in, isolated, testable, and accountable. Experimental implementations live under `flashinfer.experimental`; public functions can remain in their appropriate modules, clearly marked as experimental. An existing stable API can also expose an experimental backend.

Contributors provide reference-based correctness tests, validation on the intended hardware, and a runnable example within a declared support scope. Experimental work uses a focused CI lane; stable-core changes retain their normal review and compatibility requirements. Human-authored and agent-assisted contributions follow the same contract.

Developers control adoption. Calling an experimental API or explicitly selecting an experimental backend is an opt-in and triggers an `ExperimentalWarning` on first use. Automatic selection excludes experimental backends unless `FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1` is set.

Each feature needs an owner, a tracking issue, and a graduation plan. The default intent is graduation within four weeks; a feature may graduate, continue incubating with maintainer approval, or be removed. This is an incubation policy, not a guarantee that every feature becomes stable on a fixed date.

Choose a feature whose documented GPU, precision, shapes, and dependencies match your workload, then share correctness and performance feedback through its tracking issue.

**Read more: [Move Fast. Don’t Break Things.—Introducing FlashInfer’s Experimental Path](/2026/09/22/experimental-path.html).**

## Open attention and MoE kernels with CUTLASS Primitives and Task Scheduling

FlashInfer integrates kernels from several backends behind common APIs. The TRT-LLM Gen kernels have been distributed as prebuilt binaries, limiting developers’ ability to inspect, modify, or extend them. Prims-TS adds open-source Python implementations built on the Primitives and Task Scheduling APIs introduced in CUTLASS DSL 4.7.

Primitives provides a typed Python layer over Blackwell hardware operations, including TMA copies, tensor-core instructions, tensor memory, and barriers. Task Scheduling expresses warp-specialized kernels as tasks—such as load, MMA, online softmax, reduction, and epilogue—with declared resources and a schedule.

The attention implementation, introduced before v0.7, includes FMHA prefill, FMHA decode, and MLA decode. FlashInfer v0.7 adds fused-MoE implementations under `flashinfer.prims_ts`. Together, these implementations let developers read, debug, and modify kernel source directly, and contribute extensions through ordinary pull requests.

Prims-TS is experimental and opt-in. The kernels compile at runtime and require `nvidia-cutlass-dsl` 4.7 or newer. The existing TRT-LLM Gen APIs remain available. The Prims-TS entry points follow related API conventions and reuse routing and finalization components, but supported shapes, layouts, options, and output behavior must be checked for the specific entry point; they are not universally interchangeable with an import change alone.

Start with the [Prims-TS source and documentation](https://github.com/flashinfer-ai/flashinfer/tree/v0.7.0/flashinfer/prims_ts) and the [attention implementation](https://github.com/flashinfer-ai/flashinfer/tree/v0.7.0/flashinfer/attention/prims_ts). Compare against the existing backend on your supported configuration before modifying the kernel.

## A common interface for expert-parallel MoE

Expert-parallel MoE execution moves tokens to GPUs hosting their selected experts, computes the expert outputs, and combines the results. An inference engine also has to choose among communication and compute implementations with different topology, precision, layout, and dependency requirements.

FlashInfer’s `MoEEpLayer` provides a common entry point for split execution and MegaMoE execution. The split path separates dispatch, expert computation, and combine, allowing communication and computation backends to be composed. MegaMoE brings those operations into a fused implementation using symmetric memory, overlapping communication with computation.

Explicit backend configurations include CuTe DSL and DeepGEMM MegaMoE options. The common interface gives engine developers a consistent integration point while leaving backend requirements visible. A fused or split path still needs to be evaluated against the deployment’s token counts, topology, precision, and serving goals.

The [accompanying MegaMoE article](/2026/09/22/mega-moe.html) reports end-to-end measurements, kernel microbenchmarks, transport comparisons, and a model-quality check. Its NVFP4 configuration improves reported throughput by 1.06–1.20× on DeepSeek-V4-Flash and 1.19–1.32× on DeepSeek-V4-Pro in the tested 1×8 SM100 setup. These are comparisons between complete configurations: the NVFP4 path uses a cast checkpoint, while the baseline uses the original MX checkpoint. Read the performance tables together with their precision and evaluation conditions.

<figure>
  <img src="/assets/imgs/flashinfer-v07/moe-performance.jpg" alt="Author-reported MegaMoE prefill throughput and time-to-first-token comparisons for Mistral Large 3, Mistral Small 4, and GLM 5.2 on four GB200 GPUs.">
  <figcaption>A separate author-reported prefill experiment from vLLM PR #54049: four GB200 GPUs, 4096 input tokens and one output token, concurrency 128, TP1/DP4/EP4. The figure records the September 14, 2026 snapshot, when the PR was open and unmerged. It compares integrated backends, not isolated wrapper overhead; results were not independently reproduced and include no run-to-run uncertainty.</figcaption>
</figure>

The figure’s proposed integration is described in [vLLM PR #54049](https://github.com/vllm-project/vllm/pull/54049). Integration availability varies by model, backend, and engine version. Broader model evaluation remains ongoing.

**Read more: [MegaMoE in FlashInfer: Fused Expert-Parallel MoE Kernels](/2026/09/22/mega-moe.html).**

## Choose kernels for the way you serve with Autotuner v2

A kernel configuration that looks fastest when measuring GPU time alone can lose once the host work needed to launch it is included. CUDA Graph replay changes those costs—and can change which configuration wins.

Autotuner v2 introduces `MeasurementPolicy` to make that choice explicit. Eager measurement includes per-call host costs, while CUDA Graph measurement captures each candidate and ranks replay latency. Results measured under different policies are stored separately. The default `auto` setting preserves the previous measurement behavior.

FlashInfer also takes responsibility for organizing persistent tuning results. Hardware, software versions, and measurement policy select a compatible environment namespace; operation, runner, shape bucket, and runner-specific configuration identify an entry within it. A compatible restart can reuse saved results, while an environment change selects a separate cache. Each result is saved when tuning completes, and a damaged entry can be skipped without discarding the others. Engines retain control over warmup and worker coordination.

The untuned default competes alongside other candidates, giving the tuner a measured baseline. Selection-quality benchmarks compare the selected configuration with the fastest tested candidate under the same execution conditions; they do not guarantee the fastest kernel for every workload.

In the [accompanying article’s](/2026/09/22/autotuner-v2.html) 276 matched eager workloads, median regret—the percentage latency penalty relative to the fastest measured candidate—falls from 1.390% with v1 to 0.418% with v2. The count of workloads with more than 100% regret falls from 59 to zero. Across 281 matched CUDA Graph workloads, median regret falls from 0.0565% to 0.00646%. The benefit depends on operation and execution mode; these are selection-quality measurements, not end-to-end serving-throughput gains.

V2 is available through the opt-in `autotune_v2()` API, while the existing API remains supported. Try it during representative startup warmup, before capturing serving CUDA Graphs. Choose the measurement policy matching your execution mode, restart with the same environment to check reuse, and measure tuning time and serving performance separately.

**Read more: [FlashInfer Autotuner v2: Tune the Way You Serve](/2026/09/22/autotuner-v2.html).**

## Get started with v0.7

Review the [release highlights](/releases/#v0-7-0) for upgrade details, then use the technical posts to evaluate the features relevant to your workload. Experimental kernels provide an explicit route to early feedback, open implementations let developers extend the code, the MoE API simplifies integration, and serving-aware autotuning helps select and reuse kernel configurations.

Share reproducible correctness or performance findings through [FlashInfer’s GitHub issues](https://github.com/flashinfer-ai/flashinfer/issues), including your GPU, software versions, input shapes, backend, and measurement policy.
