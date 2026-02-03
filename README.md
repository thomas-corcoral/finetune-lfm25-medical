# Fine-tuning LFM2.5-1.2B on Medical Data

[![Python 3.8+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/TmsCorcoral/lfm25-medical-1.2b)

A complete pipeline for fine-tuning Liquid AI's **LFM2.5-1.2B-Instruct** model on medical instruction data using Unsloth for efficient training on Google Colab's free T4 GPU.

## 🎯 Project Overview

This project demonstrates Parameter-Efficient Fine-Tuning (PEFT) using LoRA to specialize a small language model for medical question-answering. The LFM2.5 model is a 1.17B parameter hybrid reasoning model that delivers best-in-class performance at the 1B scale.

### Why This Matters

- **Accessibility**: Shows how to fine-tune LLMs on free Google Colab GPUs
- **Efficiency**: Uses LoRA and 4-bit quantization to reduce memory requirements by 50%
- **Real-world Application**: Medical AI assistants can help make healthcare information more accessible
- **Best Practices**: Demonstrates proper dataset formatting, training configuration, and model export

## Quick Start

### Run on Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thomas-corcoral/finetune-lfm25-medical/blob/main/fine_tuning_LFM2_5_1_2B_on_medical_data.ipynb)

1. Click the badge above
2. Runtime → Change runtime type → T4 GPU
3. Run all cells


## Dataset

**Dataset**: [medalpaca/medical_meadow_wikidoc_patient_information](https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc_patient_information)

This dataset contains medical question-answer pairs sourced from WikiDoc, covering:
- Patient information and education
- Medical conditions and symptoms
- Treatment options and procedures
- Health and wellness topics

**Why This Dataset?**
- High-quality medical content reviewed by healthcare professionals
- Instruction-following format ideal for LLM fine-tuning
- Covers diverse medical topics
- Accessible language suitable for patient education

## Architecture

### Model: LFM2.5-1.2B-Instruct

- **Parameters**: 1.17B
- **Architecture**: 16 layers (10 double-gated LIV convolution blocks + 6 GQA blocks)
- **Training Budget**: 28T tokens
- **Context Length**: 32,768 tokens (using 2,048 for efficiency)
- **Vocabulary**: 65,536 tokens

## Key Features

### 1. Efficient Memory Usage
- **4-bit Quantization**: Reduces memory footprint by 75%
- **LoRA**: Only trains 0.5% of parameters
- **Gradient Checkpointing**: Saves memory during backpropagation
- **Fits on Free Colab T4 GPU** (15GB VRAM)

### 2. Multiple Export Formats
```python
# LoRA adapters only (~50MB)
model.save_pretrained("lfm25_medical_lora")

# Merged 16-bit model (~2.3GB)
model.save_pretrained_merged("lfm25_medical_merged", save_method="merged_16bit")

# GGUF for llama.cpp (~700MB with Q4_K_M)
model.save_pretrained_gguf("lfm25_medical_gguf", quantization_method="q4_k_m")
```

### 3. Proper Chat Formatting
Uses LFM2.5's ChatML-like format:
```
<|startoftext|><|im_start|>system
You are a knowledgeable medical assistant.<|im_end|>
<|im_start|>user
What are the symptoms of diabetes?<|im_end|>
<|im_start|>assistant
Common symptoms include...<|im_end|>
```

## Performance

### Training Results
- **Training Speed**: 2x faster than standard PyTorch with Unsloth
- **Memory Usage**: 50% less VRAM than traditional fine-tuning
- **Training Loss**: Decreases steadily (monitor in logs)

### Inference
```python
# Recommended inference settings for LFM2.5
temperature = 0.1
top_k = 50
top_p = 0.1
repetition_penalty = 1.05
```

## What You'll Learn

1. **Dataset Preparation**: Format medical Q&A data for instruction tuning
2. **LoRA Configuration**: Set up parameter-efficient fine-tuning
3. **Training Pipeline**: Use Unsloth and TRL for efficient training
4. **Model Export**: Save models in multiple formats (LoRA, merged, GGUF)
5. **Inference**: Run the fine-tuned model for medical Q&A

## Use Cases

- **Patient Education**: Provide accessible health information
- **Medical Students**: Study aid for medical concepts
- **Healthcare Apps**: Power chatbots for preliminary health questions
- **Research**: Analyze medical literature and documentation

**Important Disclaimer**: This model is for educational purposes only. Always consult qualified healthcare professionals for medical advice. Do not use for diagnosis or treatment decisions.

## Next Steps

1. **Extend Training**: Increase `max_steps` or use `num_train_epochs` for full dataset
2. **Add Validation**: Split dataset and track validation metrics
3. **Experiment with Hyperparameters**: Try different learning rates, LoRA ranks

## 📖 Resources

- [LFM2.5 Model Card](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [Medical Meadow Dataset](https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc_patient_information)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [LFM2 Technical Report](https://arxiv.org/abs/2511.23404)

## Contact

Have questions or feedback? Feel free to:
- Open an issue on GitHub
- Connect on LinkedIn

---

**If you find this project helpful, please give it a star!**

Made with code 