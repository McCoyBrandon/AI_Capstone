# AI_Capstone
# TabTransformer for Datacenter Equipment Failure Monitoring using SMOTE and Drift Detection

**Authors:** Brandon McCoy

**Date:** December 8, 2025  

---
## Directory
Additional terminal usage notes on top of relevant files.

AI_Capstone/
├─ .github/
│  └─ ci.yaml
│
├─ configs/
│  └─ train.yaml
│
├─ runs/
│  └─ main_save/
│     ├─ figs/
│     │  ├─ confusion_matrix_focused_epoch40.png
│     │  ├─ pre_post_smote_counts.png
│     │  ├─ roc_pr_curves_epoch40.png
│     │  ├─ train_lables_post_bar.png
│     │  └─ tb/          
│     ├─ metrics.json
│     └─ model.ckpt      # Final model checkpoint
│
├─ docs/
│  ├─ mccoy_poster_presentation.pptx
│  ├─ mccoy_poster_presentation.pdf
│  └─ Final_Techinical_Report.pdf
│
├─ src/
│  ├─ data/
│  │  ├─ ai4i2020.csv
│  │  ├─ data_loader.py
│  │  └─ preprocess.py
│  │
│  ├─ deploy/
│  │  ├─ infer.py         
│  │  └─ demo/
│  │     ├─ demo.py
│  │     └─ Huggingface/
│  │        ├─ app.py
│  │        ├─ README.md
│  │        └─ requirement.txt
│  │
│  ├─ models/
│  │  └─ transformer_class.py   # TinyTabTransformer architecture
│  │
│  ├─ training/
│  │  ├─ train.py          # Main trainer
│  │  ├─ grid_search_train.py
│  │  └─ eval.py           # Evaluation script
│  │
│  └─ utils/
│     ├─ check_grid_search.py
│     ├─ metrics.py
│     ├─ metrics_viz.py
│     └─ visualizations.py
│
├─ .dockerignore
├─ .gitignore
├─ docker-test.yml
├─ dockerfile
├─ dvc.yaml
├─ makefile
├─ requirements.txt
└─ README.md

---

## Interactive Demo (Hugging Face)

An interactive version of this project is hosted as a Hugging Face Space, which allows you to test the TabTransformer model directly in your browser without installing anything locally.

**Live Demo:**  
https://huggingface.co/spaces/mccoybs/Machine_Failure_Prediction_Demo

In the demo you can:
- Adjust feature values (machine type, temperature, torque, etc.)
- Perform inference with the latest trained model
- View predicted failure type and individual class probabilities

---

## How to reproduce usking Windows Command Prompt or PowerShell pre-organized MAKEFILE set up.

These steps fully reproduce the project end-to-end in a fresh environment.

---

### 1. Clone & install dependencies

```bash
git clone https://github.com/McCoyBrandon/AI_Capstone
cd AI_Capstone

python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dataset 
Ensure that the dataset is located in the data folder:
```bash
AI_Capstone/src/data/ai4i2020.csv
```

If not, you can download the dataset here:
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

### 3. Train
Training uses a single command and saves logs + checkpoint under:
AI_Capstone/runs/Test_run_1

```bash
python -m src.training.train
```

### 4. Evaluate a trained model

Evaluate the model trained in step above:
```bash
python -m src.training.eval `
  --ckpt runs/Test_run_1/model.ckpt `
  --out  runs/Test_run_1/eval_metrics.json
```

This loads from the checkpoint:

runs/Test_run_1/model.ckpt

and writes results to:

runs/Test_run_1/eval_metrics.json

# 5. Optional: Inference demo
To use the above checkpoint in an inference:
```bash
python -m src.deploy.infer `
  --ckpt runs/Test_run_1/model.ckpt `
  --csv  src/data/ai4i2020.csv `
  --out  runs/Test_run_1/demo_metrics.json `
  --failures-csv runs/Test_run_1/flagged_failures.csv
```

Which will output:
runs/Test_run_1/demo_metrics.json
runs/Test_run_1/flagged_failure.csv

# 6. End-to-end summary:
In order to demo the use of this repository and produce your own:
```bash
git clone https://github.com/McCoyBrandon/AI_Capstone
cd AI_Capstone
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

python -m src.training.train
python -m src.training.eval --ckpt runs/Test_run_1/model.ckpt --out runs/Test_run_1/eval_metrics.json
python -m src.deploy.infer --ckpt runs/Test_run_1/model.ckpt --csv src/data/ai4i2020.csv --out runs/Test_run_1/demo_metrics.json
```
---
## How to reproduce usking Docker.
The same end-to-end pipeline (train → eval → infer) can be run inside a Docker container.

### 1. Build the image

From the repo root:

```bash
docker build -t tabtransformer-ai4i .
```

### 2. Train the model

The default command in the image runs training with run_name = Test_run_1.
To train and keep outputs on your host machine:

This will:

- load src/data/ai4i2020.csv inside the container

- preprocess (normalize + SMOTE)

- train the TinyTabTransformer

- write logs/checkpoints/figures into runs/Test_run_1/ 

### 2. Evaluate the trained model

Use the same image, but override the command to run the eval script:

```bash
docker run --rm \
  -v "$(pwd)/runs:/app/runs" \
  tabtransformer-ai4i \
  python -m src.training.eval \
    --ckpt runs/Test_run_1/model.ckpt \
    --out  runs/Test_run_1/eval_metrics.json
```
This will:

- mounts your runs/ directory into the container

- loads runs/Test_run_1/model.ckpt

- writes runs/Test_run_1/eval_metrics.json

### 3. Run inference demo
Similar to to above, for the inference you will run:

```bash
docker run --rm \
  -v "$(pwd)/runs:/app/runs" \
  -v "$(pwd)/src/data:/app/src/data" \
  tabtransformer-ai4i \
  python -m src.deploy.infer \
    --ckpt runs/Test_run_1/model.ckpt \
    --csv  src/data/ai4i2020.csv \
    --out  runs/Test_run_1/demo_metrics.json \
    --failures-csv runs/Test_run_1/flagged_failures.csv
```
Here we also mount src/data so the container can see ai4i2020.csv.

### Docker summar:
```bash
docker build -t tabtransformer-ai4i .
docker run --rm -v "$(pwd)/runs:/app/runs" tabtransformer-ai4i
docker run --rm -v "$(pwd)/runs:/app/runs" tabtransformer-ai4i \
  python -m src.training.eval \
    --ckpt runs/Test_run_1/model.ckpt \
    --out  runs/Test_run_1/eval_metrics.json
docker run --rm \
  -v "$(pwd)/runs:/app/runs" \
  -v "$(pwd)/src/data:/app/src/data" \
  tabtransformer-ai4i \
  python -m src.deploy.infer \
    --ckpt runs/Test_run_1/model.ckpt \
    --csv  src/data/ai4i2020.csv \
    --out  runs/Test_run_1/demo_metrics.json \
    --failures-csv runs/Test_run_1/flagged_failures.csv
```
This mirrors the non-Docker workflow:

- make train → train in container

- make eval → eval in container (with command override)

- make infer → infer in container (with command override)

---

## Introduction

Datacenter downtimes can cause huge financial losses and reputational damage. Notable examples include in 2017 when Amazon's S3 service was down for a few hours with an estimate of roughly $300 million in lost revenue for S&P 500 and U.S. financial-service companies. Another being in Australia in 2023 with the Optus outage that lasted about 11–12 hours. Optus is widely used by the government and private sectors, resulting in crucial services in public safety and logistics causing failed communications and heavy delays. Ultimately costing the company A$2 billion in stock, senate inquiry, and large reputation lost.

These downtimes can often be a result of cascading failures, in which one problem may spark a domino effect in poorly managed systems that results in a catastrophic failure incident. But having humans track second-by-second diagnostics for a datacenter with thousands of machines on a 24/7 basis can be expensive and may cost as much as the potential downtimes themselves. The need for a proactive machine failure tracking system that can be run by a small team and is more cost-efficient is increasingly important in the modern information age and cloud computing world.

---

## Problem Definition

Given the diagnostic data in a tabular format for devices in a datacenter, the objective is to create a transformer that utilizes attention to accurately track for the risk of devices failing. While also addressing issues of class imbalance due to low classification counts in comparison to the overall data. The classification should come in two forms, firstly a binary ‘True’ or ‘False’ for risk of machine failure, and secondly a multi-class estimate for the type of failure. This includes tool wear failure (TWF), heat dissipation failure (HDF), power failure (PWF), and overstrain failure (OSF).  Additionally, there needs to be an output able to specify the machine ID that is being flagged with probability predicted of failure.  Ideally the prediction model should be able to run on a lightweight device.


---
## Relevant work



---

## Methodology

This project implements a machine-learning pipeline designed to predict machine failure events and classify specific failure types using tabular diagnostic data from datacenter equipment. The workflow begins with extensive data preprocessing, including handling missing values, standardizing numerical features, and encoding categorical machine types. Because failure events are extremely rare relative to normal operating observations, the dataset exhibited a substantial class imbalance. Which would otherwise cause traditional models to overfit to the majority class and overlook critical minority patterns. To address this, the SMOTE (Synthetic Minority Oversampling Technique) algorithm was applied to generate realistic synthetic examples of under-represented failure categories. By interpolating between existing minority samples in feature space rather than duplicating records, SMOTE improves the density of minority regions and supports the learning of more effective classification boundaries.

The core predictive model utilizes a TabTransformer architecture, which leverages multi-head self-attention to learn contextual relationships between tabular features. This approach allows the network to capture dependencies and interactions that linear and tree-based models may miss. The data was divided into training, validation, and separated test sets to reflect chronological behavior and prevent performance leakage. Because machine behavior and sensor distributions can evolve over time, a drift detection module was incorporated to monitor statistical distribution shifts and performance decay during inference. This could be could also be used to created indicator for model retraining or pipeline recalibration before significant degradation occurs.

Model performance was evaluated using macro-F1 score, recall, precision, accuracy, and AUC-ROC, with additional focus on confusion matrix analysis to verify improvements in failure type classification sensitivity. Metrics were benchmarked both before and after SMOTE augmentation to quantify its impact. This integrated methodology supports early and reliable identification of high-risk machine conditions, which could be used by IT teams to enable proactive maintenance strategies and reducing the likelihood of unplanned downtime in large-scale datacenter environments.

---

## Things of note

First and foremost, this project has been a great way to learn about how to deal with datasets that use continuous data that has a large amount of observations for ‘normal’ states (NoFailure for AI4I data) and a small amount of targeted events, such as machine failures for a datacenter with a large number of machines. The binary classifier of ‘Failure’ vs ‘NoFailure’ is reasonably achievable as it uses the sum of all the sub-categories of failures. But when getting granular with the failure types, using traditional class balancing technique such as weighting the class split for the train/text/validation datasets will not suffice.

This is where the Synthetic Minority Oversampling Technique (SMOTE) came in very helpful. As it does not simply create duplicate records for the underrepresented classes, but instead uses a nearest neighbor method of generating synthetic observations for the underrepresented multi-class values.  This is then extrapolated out until the underrepresented failure types have equal representation between ‘NoFailure’ and each type of failure.

As for the model selection, while reading the literature I found most examples of working with this dataset utilize decision trees, SVM, XGBoost, and ensemble methods. With decision trees being the most effective, however often having an overfitting problem. And after reading ‘Attentional is all you need’ I wanted to see how a transformer utilizing attention faired in comparison. That is where I found the TabTransformer and used it as my baseline architecture.  My model didn’t perform as well as those models claimed, but I also didn’t have extra datasets for training and didn’t to explore continued learning and utilization of the drift detection for hyper tuning.  Which are possible developments for the future and may be able to close the accuracy gap with less risk of overfitting.
One of the concerns about using the SMOTE method is that real world changes in equipment or run-times may result in changing conditions in which the prediction effectiveness degrades, or original training data doesn’t fully represent real scenarios. That is why drift detection was additionally important in exploring. For this iteration it allows to diagnosis for model performance, but in future developments of continued learning it could be utilize to tune the model.

---
