# FlashSinkhorn / FlashWasserstein Latent OT-CFM

This temporary repo packages the code needed to run the latent ImageNet
OT-CFM experiments on an H100 node.

Important paths:

- `code/`: FlashSinkhorn package and Triton kernels.
- `FlashWasserstein/`: FlashWasserstein prototype and benchmarks.
- `FlashWasserstein/conditional-flow-matching-main/`: OT-CFM fork.
- `FlashWasserstein/conditional-flow-matching-main/examples/images/latent_imagenet/`: latent ImageNet OT-CFM experiment.
- `H100_RUNBOOK.md`: exact H100 setup, encode, benchmark, train, and sampling commands.

Generated outputs, cached latents, local envs, and large media artifacts are not
tracked in git.
