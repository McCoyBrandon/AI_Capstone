.PHONY: train eval infer

train:
	python -m src.training.train --run_name Test_run_1

eval:
	python -m src.training.eval --ckpt runs/Test_run_1/model.ckpt --out  runs/Test_run_1/eval_metrics.json

infer:
	python -m src.deploy.infer --ckpt runs/Test_run_1/model.ckpt --csv  src/data/ai4i2020.csv --out  runs/Test_run_1/demo_metrics.json --failures-csv runs/Test_run_1/flagged_failures.csv