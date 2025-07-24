#!/usr/bin/env python3
"""
Script to download and setup LJSpeech dataset for HiFiGAN training on macOS
"""

import os
import urllib.request
import tarfile
from pathlib import Path

def download_ljspeech():
    """Download and extract LJSpeech dataset."""
    
    dataset_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    dataset_file = "LJSpeech-1.1.tar.bz2"
    
    print("Downloading LJSpeech dataset...")
    print("This may take a while (approximately 2.6GB)")
    
    # Download with progress
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = (downloaded / total_size) * 100
        print(f"\rProgress: {percent:.1f}% ({downloaded // (1024*1024):.1f}MB / {total_size // (1024*1024):.1f}MB)", end="")
    
    try:
        urllib.request.urlretrieve(dataset_url, dataset_file, show_progress)
        print("\nDownload completed!")
        
        print("Extracting dataset...")
        with tarfile.open(dataset_file, 'r:bz2') as tar:
            tar.extractall()
        
        print("Dataset extracted successfully!")
        
        # Clean up
        os.remove(dataset_file)
        print("Temporary files cleaned up.")
        
        # Verify extraction
        wavs_dir = Path("LJSpeech-1.1/wavs")
        if wavs_dir.exists():
            num_wavs = len(list(wavs_dir.glob("*.wav")))
            print(f"Found {num_wavs} audio files in {wavs_dir}")
            print("Dataset setup completed successfully!")
            return True
        else:
            print("Error: wavs directory not found after extraction")
            return False
            
    except Exception as e:
        print(f"Error during download/extraction: {e}")
        return False

if __name__ == "__main__":
    if download_ljspeech():
        print("\n" + "="*50)
        print("Dataset setup complete!")
        print("You can now run training with:")
        print("python train.py --config config_v1.json")
        print("="*50)
    else:
        print("Dataset setup failed. Please check your internet connection and try again.")
