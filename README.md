# AI_Capstone
# On-Device Transformer for Datacenter Monitoring - Design Document

**Authors:** Brandon McCoy and Venugopal Ponnamaneni  
**Date:** September 5, 2025  

---

## Introduction
Datacenter downtimes can cause huge financial losses. Having humans track second-by-second diagnostics for a datacenter with thousands of machines on a 24/7 basis can be expensive and may cost as much as the potential downtimes themselves. The need for a tracking system that can be run by a small team and is more cost-efficient is increasingly important in the modern world.

This system can benefit datacenter administrators by enabling preemptive preventive maintenance, avoiding cascading failures, and maximizing the lifetime of expensive assets.  

The expectation is that our transformer can run on a lightweight system that flags possible failure risk factors. It will operate in real time and flag potential failures for a system administrator to address and plan around.

---

## Functional Requirements
The system focuses on applying lightweight transformer models for predictive maintenance using the **AI4I 2020 dataset**. This dataset contains both numerical and categorical variables related to machine operating conditions such as temperature, rotational speed, torque, and tool wear. These inputs will be preprocessed and fed into the model, which is expected to classify whether a machine is at risk of failure.  

Key requirements:
- Minimum accuracy of **0.7** on the testing set (target: **0.8+**).  
- Predictions are **binary** (failure or no failure).  
- **Real-time inference** with latency under **1 second**.  
- All predictions logged with timestamps for review.  
- Deployment target: **NVIDIA Orin** or **Raspberry Pi**.  
- Transformer size around **125 MB** to fit device constraints.  
- Framework: **PyTorch**, with **TensorBoard** for visualization and **scikit-learn** preprocessing.  
- A **custom-built model** (no pretrained available for this dataset).  

---

## Non-Functional Requirements
The non-functional requirements define quality standards for system behavior.  

- **Reliability:** Must provide consistent and trustworthy results. Evaluations will include test subsets for stability and repeatability.  
- **Deployment:** Initially static (no retraining). Future versions may include **continuous learning** with new telemetry data.  
- **Access Control:** Prediction results and telemetry restricted to system administrators. Real-world use may extend to **anonymization, encryption, and secure logging**.  
- **Efficiency:** Must run effectively on constrained devices without exceeding CPU/memory budgets.  
- **Interpretability:** Explore tools such as **LIME** for transparency, compression, and usability.  

---

## System Design
The proposed system is designed for **on-device inference** using compact hardware (NVIDIA Orin, Raspberry Pi). This eliminates dependency on cloud servers while balancing model complexity with efficiency.  

Core design elements:
- **Lightweight transformer (~125 MB)** implemented in PyTorch.  
- Input preprocessing with encoding and scaling (via scikit-learn).  
- **TensorBoard** for monitoring training and logging.  
- Modular design to allow future extensions (e.g., encryption/anonymization).  
- Primary objective: **accurate, efficient, real-time predictions** for datacenter machine health.  

---

## Testing / Validation Plan
Testing will use the **AI4I 2020 Predictive Maintenance Dataset**, split into training/validation/testing (70/15/15).  

Validation steps:
- Ensure subsets represent balanced machine types and values.  
- Binary classification evaluation (failure vs. no failure).  
  - **Accuracy threshold:** <0.7 = fail, ≥0.8 = target success.  
- If binary classification is satisfactory, extend to **multi-class failure prediction** using **AUROC metrics**.  
- Measure **latency** (end-to-end) and **CPU utilization** (average/peak).  
- Test on both **virtual machines** and target hardware (NVIDIA Orin or Raspberry Pi).  

---

## Contribution Statement
**Brandon McCoy**  
- Note-taking and tracking requirements for the outline.  
- Wrote the **Introduction** and **Testing/Validation Plan** sections.  

**Venugopal Ponnamaneni**  
- Expanded the outline into the **Functional Requirements**, **Non-Functional Requirements**, and **System Design** sections.  

---
