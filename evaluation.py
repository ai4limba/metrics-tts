"""
evaluation.py

This script evaluates synthesized speech samples using the DNSMOSPro and SQUIM objective quality metrics.
It outputs a CSV file with average and standard deviation metrics per model variant.
"""

import os
import sys
import tqdm # For progress bar
import torch
import librosa
import argparse
import numpy as np
import pandas as pd

import tensorflow as tf
import DNSMOSPro.utils as utils

from dotenv import load_dotenv
from torchaudio.pipelines import SQUIM_OBJECTIVE

# Not every audio sample rate is at 16kHz, so we resample them to this value in order to obtain correct and fair values
# this is advised by the DNSMOSPro documentation; we also pad or truncate the audio to 10 seconds
TARGET_SR       = 16_000 
TARGET_SECONDS  = 10
TARGET_SAMPLES  = TARGET_SR * TARGET_SECONDS

def preprocess_audio_to_image(audio, 
                              sr=24000, 
                              n_mels=128,
                              win_length=1024, 
                              hop_length=512, 
                              target_size=(224, 224),
                              eps=1e-6):

    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=win_length,
        hop_length=hop_length, n_mels=n_mels
    )

    mel_spec = np.maximum(mel_spec, eps)


    log_mel_spec = librosa.power_to_db(
        mel_spec, ref=np.max, top_db=80
    )


    finite_vals = log_mel_spec[np.isfinite(log_mel_spec)]
    if finite_vals.size:
        floor = finite_vals.min()
        log_mel_spec = np.where(np.isfinite(log_mel_spec),
                                log_mel_spec,
                                floor)
    else:
        log_mel_spec = np.zeros_like(log_mel_spec)

    min_val = log_mel_spec.min()
    max_val = log_mel_spec.max()
    denom = max_val - min_val
    if denom > 0:
        log_mel_spec_norm = (log_mel_spec - min_val) / denom
    else:
        log_mel_spec_norm = np.zeros_like(log_mel_spec)

    img = np.stack([log_mel_spec_norm]*3, axis=-1)
    img_tensor = tf.image.resize(img, target_size)
    return img_tensor.numpy()

def load_models(dnsmos_path: str, srdorigin_path: str):
    """Load DNSMOSPro and SQUIM models."""
    if not os.path.exists(dnsmos_path):
        sys.exit(f"ERROR: DNSMOSPro model not found at: {dnsmos_path}")
        
    try:
        DNSMOS_model = torch.jit.load(dnsmos_path, map_location=torch.device('cpu'))
    except Exception as e:
        sys.exit(f"ERROR: Failed to load DNSMOSPro model. Error: {e}")
        
    try:
        SQUIM_model = SQUIM_OBJECTIVE.get_model()
    except Exception as e:
        sys.exit(f"ERROR: Failed to load SQUIM model. Error: {e}")
        
    try:
        SRDORIGIN_model = tf.keras.models.load_model(srdorigin_path)
    except Exception as e:
        sys.exit(f"ERROR: Failed to load SRDORIGIN model. Error: {e}")
        
    return DNSMOS_model, SQUIM_model, SRDORIGIN_model

def preprocess_audio(filepath: str):
    """Load and preprocess audio file for evaluation."""
    
    wav, sr = librosa.load(filepath)        
    if sr != TARGET_SR: # I resample the audio only if necessary
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        
    # Ensure the audio is of the correct length
    if len(wav) < TARGET_SAMPLES:
        pad_amt = TARGET_SAMPLES - len(wav)
        wav = np.pad(wav, (0, pad_amt), mode='constant')
    else:
        wav = wav[:TARGET_SAMPLES]
    
    return wav

def evaluate_sample(wav, dnsmos_model, squim_model, srdorigin_model):
    """Evaluate a single audio sample using DNSMOSPro and SQUIM"""
    
    SQUIM_wav = torch.tensor(wav)
    if SQUIM_wav.dim() == 1:
        SQUIM_wav = SQUIM_wav.unsqueeze(0)
        
    spec = torch.FloatTensor(utils.stft(wav))
    
    with torch.no_grad():
        prediction = dnsmos_model(spec[None, None, ...])
        stoi, pesq, si_sdr = squim_model(SQUIM_wav)
        
    img = preprocess_audio_to_image(wav)
    x = np.expand_dims(img, axis=0)
    
    srdorg = srdorigin_model.predict(x, verbose=0)
        
    mos_mean = prediction[:, 0].item()
    stoi = stoi.item()
    pesq = pesq.item()
    si_sdr = si_sdr.item()
    srdorg = srdorg[0][0]
    
    return mos_mean, stoi, pesq, si_sdr, srdorg

def evaluate_folder(root_dir, folder_name, dnsmos_model, squim_model, srdorigin_model):
    """Evaluate all samples in a folder."""
    
    path = os.path.join(root_dir, folder_name)
    if not os.path.isdir(path):
        print(f"WARNING: Folder {folder_name} does not exist at path: {path}. Skipping this folder.")
        return None
    
    dnsmos_scores, stoi_scores, pesq_scores, sisdr_scores, srd_origin = [], [], [], [], []
    
    for sample in tqdm.tqdm(os.listdir(path), desc=f"Evaluating {folder_name}"):
        sample_dir = os.path.join(path, sample)
        if not os.path.isdir(sample_dir):
            print(f"WARNING: Sample {sample} is not a directory. Skipping this sample.")
            continue
        
        sample_path = os.path.join(sample_dir, "gen.wav")
        
        if not os.path.exists(sample_path):
            print(f"WARNING: Sample {sample} does not have a 'gen.wav' file at path: {sample_path}. Skipping this sample.")
            continue
        
        wav = preprocess_audio(sample_path)
        
        try: 
            dnsmos, stoi, pesq, sisdr, srdorg = evaluate_sample(wav, dnsmos_model, squim_model, srdorigin_model)
            dnsmos_scores.append(dnsmos)
            if not np.isnan(stoi): stoi_scores.append(stoi)
            if not np.isnan(pesq): pesq_scores.append(pesq)
            if not np.isnan(sisdr): sisdr_scores.append(sisdr)
            if not np.isnan(srdorg): srd_origin.append(srdorg)
        except Exception as e:
            print(f"ERROR: Failed to evaluate sample {sample} in folder {folder_name}. Error: {e}")
            continue
    
    if len(dnsmos_scores) == 0:
        print(f"WARNING: No valid samples found in folder {folder_name}.")
        return None
    
    # SNR - Signal Noise Ratio & BER - Bit Error Rate if computed
    if folder_name.split("-")[-1] == "Watermarked": # if the folder is a watermarked one then we load the watermarking results
        watermarking_results = pd.read_csv(os.path.join(root_dir, "watermarking_results.csv"))
        
        # Extract the SNR and BER values for the current folder
        snr_std = watermarking_results.loc[watermarking_results['Folder'] == "-".join(folder_name.split("-")[:-1]), 'SNR - Std'].values
        snr_mean = watermarking_results.loc[watermarking_results['Folder'] == "-".join(folder_name.split("-")[:-1]), 'SNR - Avg'].values
        ber_std = watermarking_results.loc[watermarking_results['Folder'] == "-".join(folder_name.split("-")[:-1]), 'BER - Std'].values
        ber_mean = watermarking_results.loc[watermarking_results['Folder'] == "-".join(folder_name.split("-")[:-1]), 'BER - Avg'].values
    else:
        snr_std = None
        snr_mean = None
        ber_std = None
        ber_mean = None
    
    return {
        "ModelName": "-".join(folder_name.split("-")[1:]),
        "DNSMOSPro - Std": round(np.std(dnsmos_scores), 2),
        "DNSMOSPro - Avg": round(np.mean(dnsmos_scores), 2),
        "PESQ - Std": round(np.std(pesq_scores), 2),
        "PESQ - Avg": round(np.mean(pesq_scores), 2),
        "STOI - Std": round(np.std(stoi_scores), 2),
        "STOI - Avg": round(np.mean(stoi_scores), 2),
        "SI-SDR - Std": round(np.std(sisdr_scores), 2),
        "SI-SDR - Avg": round(np.mean(sisdr_scores), 2),
        "SrdOrigin - Std": round(np.std(srd_origin), 2),
        "SrdOrigin - Avg": round(np.mean(srd_origin), 2),
        "SNR - Std": round(snr_std[0], 2) if snr_std is not None else None,
        "SNR - Avg": round(snr_mean[0], 2) if snr_mean is not None else None,
        "BER - Std": round(ber_std[0], 2) if ber_std is not None else None,
        "BER - Avg": round(ber_mean[0], 2) if ber_mean is not None else None
    } 
        

def main():
    parser = argparse.ArgumentParser(description="Evaluate synthesized speech samples using DNSMOSPro and SQUIM")
    parser.add_argument('--root_dir', type=str, required=False, help="Path to root directory with samples.")
    parser.add_argument('--output_csv', type=str, default='evaluation_results.csv', help="Output CSV file for evaluation results.")
    parser.add_argument('--dnsmos_model_path', type=str, default='DNSMOSPro/runs/NISQA/model_best.pt', help="Path to DNSMOSPro model file.")
    parser.add_argument('--srdorigin_model_path', type=str, default='srdorigin_models/best_model.h5', help="Path to DNSMOSPro model file.")
    
    args = parser.parse_args()
    
    # Load the env variables
    load_dotenv()
    
    # load the root directory from the command line argument or from the environment variable
    ROOT_DIR = args.root_dir or os.getenv('ROOT_DIR')
    if ROOT_DIR is None:
        sys.exit("ERROR: ROOT_DIR environment variable not set in .env file.")
        
    # Load DNSMOSPro and SQUIM models
    DNSMOSPro_model, SQUIM_model, SRDORIGIN_model = load_models(args.dnsmos_model_path, args.srdorigin_model_path)
    
    # These are the folder names of the samples we want to evaluate
    samples_folders = [
        # "Samples-XTTSv2-Base-Mannigos",      # non-finetuned XTTSv2 synthesized audios               (Mannigos dataset)
        # "Samples-XTTSv2-Mannigos-v2",        # Mannigos-finetuned XTTSv2 synthesized audios          (Mannigos dataset)
        # "Samples-F5TTS-Base-Mannigos",       # non-finetuned F5TTS synthesized audios                (Mannigos dataset)
        # "Samples-F5TTS-Mannigos-v1",         # Mannigos-finetuned F5TTS synthesized audios          (Mannigos dataset)
        # "Samples-XTTSv2-Base-SardinianVox",  # non-finetuned XTTSv2 synthesized audios               (SardinianVox dataset)
        # "Samples-XTTSv2-SardinianVox-v2",     # SardinianVox-finetuned XTTSv2 synthesized audios      (SardinianVox dataset)
        # "Samples-F5TTS-Base-SardinianVox",   # non-finetuned F5TTS synthesized audios                (SardinianVox dataset)
        # "Samples-F5TTS-SardinianVox",        # SardinianVox-finetuned F5TTS synthesized audios       (SardinianVox dataset)
        # Watermarks
        "Samples-XTTSv2-Base-Mannigos-Watermarked",      
        "Samples-XTTSv2-Mannigos-v2-Watermarked",        
        "Samples-F5TTS-Base-Mannigos-Watermarked",       
        "Samples-F5TTS-Mannigos-v1-Watermarked",         
        "Samples-XTTSv2-Base-SardinianVox-Watermarked",  
        "Samples-XTTSv2-SardinianVox-v2-Watermarked",    
        "Samples-F5TTS-Base-SardinianVox-Watermarked",   
        "Samples-F5TTS-SardinianVox-Watermarked",        
    ]
    
    results = []
    
    for folder in tqdm.tqdm(samples_folders, desc="Folders"):
        metrics = evaluate_folder(ROOT_DIR, folder, DNSMOSPro_model, SQUIM_model, SRDORIGIN_model)
        if metrics:
            results.append(metrics)
            
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print("Evaluation completed. Results saved to:", args.output_csv)


if __name__ == "__main__":
    main()