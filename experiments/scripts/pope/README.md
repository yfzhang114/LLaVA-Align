# Experiment Scripts

This directory contains scripts to reproduce specific experiments and generate figures presented in the paper.

## COCO Identities Bias Analysis (Figures 1 and 10)

To generate figures depicting LLm biases on COCO identities, run:

```bash
sh ./scripts/pope/llava1.5_coco_bias.bash
```

## Naive and VCD Experiments
To run experiments for naive and VCD (Vision Contrastive Decoding) baselines, use:

```bash
sh ./scripts/pope/llava1.5_baselines.bash
```

For evaluation

```bash
python eval/eval_pope.py --gt_files ./data/POPE/coco/coco_pope_{split}.json --gen_files your_generative_file.json
```

## Main Results (Qwen Models, Llava Models, and LLaVA-SFT Model)

For reproducing the main results in the paper, including Qwen models, Llava models, and the LLaVA-SFT model, execute:

```bash
sh ./scripts/pope/run_dataset.bash
```

For evaluation

```bash
python eval/eval_pope_calibrate.py --gt_files ./data/POPE/coco/coco_pope_{split}.json --gen_files your_generative_file.json
```

Feel free to modify these scripts or refer to the documentation in each script for additional details on experiment configurations and parameters.