# Epilepsy Detection Using TUH Dataset

This repository contains code and resources for developing a machine learning model to detect epilepsy from EEG (electroencephalogram) signals. The dataset used for training and evaluation is the [TUH EEG Seizure Corpus](https://www.isip.piconepress.com/projects/tuh_eeg/html/), a large collection of EEG recordings from patients with epilepsy.

## Project Overview

The goal of this project is to create a reliable and efficient model to automatically detect seizures and classify EEG data from the TUH dataset. The model will help clinicians and healthcare providers in identifying seizures in real-time, aiding in early detection and treatment of epilepsy.

### Files in the Dataset:
- EEG data files in .edf foramt

## Installation

To get started with this project, follow these steps:
### 1. Get access to data from TUH:
https://isip.piconepress.com/projects/nedc/html/tuh_eeg/

### 2. Clone the Repository:

git clone https://github.com/yourusername/epilepsy_detection_TUH.git
<!-- downolad data as described on TUH page -->
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 model.py
