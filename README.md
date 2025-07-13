# metrics-tts

**metrics-tts** is a toolkit for evaluating synthesized speech using multiple quality metrics, including:
- [DNSMOSPro](https://github.com/fcumlin/DNSMOSPro)
- [SQUIM](https://docs.pytorch.org/audio/main/tutorials/squim_tutorial.html)
- **SrdOrigin**, a custom ResNet-based classifier that detects if a sample originates from Sardinian speech.

It is designed to benchmark and compare different TTS model outputs using both standard and custom quality metrics.

---

## Project Structure

- `evaluation.py`: evaluates audio samples from various model outputs across DNSMOSPro, SQUIM, and SrdOrigin.
- `srdorigin.py`: trains a ResNet152V2-based classifier to detect Sardinian speech samples.
- `srdorigin_models/`: expected folder for storing the trained SrdOrigin model
- `.env`: used to load custom paths for dataset and samples (see below).

---

## Installation
All the tests have been done with python 3.10

First, we clone the needed repositories
```bash
git clone https://github.com/ai4limba/metrics-tts
cd metrics-tts
git clone https://github.com/fcumlin/DNSMOSPro
```

Let's start with installing the DNSMOSPro requirements:
```bash
cd DNSMOSPro
pip install -r requirements.txt
pip install torch==2.1.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```
We proceed with installing our requirements:
```bash
cd .. 
pip install -r requirements.txt
```