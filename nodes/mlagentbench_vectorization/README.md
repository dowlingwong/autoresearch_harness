# mlagentbench_vectorization

This node is a bounded adapter for the MLAgentBench `vectorization` task.
The original task asks an agent to speed up a NumPy convolution forward pass.
For this harness node, the code and correctness check are frozen and the
manager may only edit `config.yaml`.

Source benchmark:

- Repository: https://github.com/snap-stanford/MLAgentBench
- Task: `MLAgentBench/benchmarks/vectorization`
- Source revision inspected for this adapter: `5d71205cc20a8e95d43aa7cb7120e89ca3323e31`
- License in source repository: MIT

The governed metric is `speed_score = 1 / median_runtime_seconds`; higher is
better. Runtime is also logged for auditability.
