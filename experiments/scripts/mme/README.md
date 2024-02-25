# Experiment Scripts

This directory contains scripts to reproduce specific experiments considering LLaVA-Bench


## Main Results (Qwen Models, Llava Models, and LLaVA-SFT Model)

For reproducing the main results in the paper, including Qwen models, Llava models, and the LLaVA-SFT model, execute:

```bash
sh ./scripts/mme/run_llava.sh # for LLaVA models
sh ./scripts/mme/run_qwen.sh # for Qwen models
```

For evaluation

```bash
python eval/MME/convert_answer_to_mme.py --experiment your_output_file_name
cd eval_tool
python calculation.py --results_dir answers/your_output_file_name
```

For evaluating post-hoc debias methods

```bash
python eval/MME/convert_answer_to_mme_calibrate.py --experiment your_output_file_name

cd eval_tool
python calculation_calibrate.py --results_dir answers/your_output_file_name
```

Feel free to modify these scripts or refer to the documentation in each script for additional details on experiment configurations and parameters.