# Experiment Scripts

This directory contains scripts to reproduce specific experiments considering LLaVA-Bench


## Main Results (Qwen Models, Llava Models, and LLaVA-SFT Model)

For reproducing the main results in the paper, including Qwen models, Llava models, and the LLaVA-SFT model, execute:

```bash
sh ./scripts/llava_bench/llava_bench.bash
```

For evaluation

```bash
OPENAI_API_KEY="" python ./eval/eval_gpt_review_visual.py \
    --question ./data/qa90_questions.jsonl \
    --context ./data/caps_boxes_coco2014_val_80.jsonl \
    --answer-list \
    ./data/qa90_gpt4_answer.jsonl \
    ./output/llava_bench/13B/llava_bench_vdd_seed${seed}_default.jsonl \
    --rule ./data/rule.json \
    --output output/llava_bench_results/review-file-qwen_naive_seed53_default.jsonl
```

```bash
python eval/summarize_gpt_review.py --dir output/llava_bench_results/
```

Feel free to modify these scripts or refer to the documentation in each script for additional details on experiment configurations and parameters.