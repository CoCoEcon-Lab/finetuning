## Model and Training Limits

All workshop exercises run on **$\geq$ 16 GB NVIDIA GPUs**, so the training configuration must fit comfortably within this memory budget.

For the fine-tuning exercises we will use:

Model: [**HuggingFaceTB/SmolLM2-360M-Instruct**](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)

This model was chosen because it runs reliably on 16 GB GPUs while still allowing meaningful experimentation with fine-tuning hyperparameter settings (rank, batching geometry, target layers)

### Allowed configurations

For **LoRA** fine-tuning, you are free to use:

- **LoRA rank:** $2-32$
- **Target layers:** attention, MLP or both
- **Global batch size:** $\leq 4$

For **RL** fine-tuning, you are free to use:

- TODO!

*(Global batch size = `micro_batch_size × gradient_accumulation_steps`)*

This setup keeps GPU memory usage safely within limits while still giving you flexibility to explore different LoRA configurations.