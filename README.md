# Injury Classification using Hybrid LSTM-Transformer Model

## Program Introduction

The present project employs a deep learning model to analyze the physiological indices of emergency department patients and stratifies them into five injury severity levels:
- 0: Healthy
- 1: Mild
- 2: Moderate
- 3: Severe
- 4: Extremely Severe

## Project Structure

```
injury_classification/
├── config.py                 
├── data                     
├── src/                      
│   ├── data_loader.py        
│   ├── feature_extraction.py 
│   ├── models.py             
│   ├── losses.py             
│   ├── training.py           
│   └── evaluation.py         
├── scripts/                  
│   └── train.py              
├── results                
├── requirements.txt          
└── README.md                 
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train Model

```bash
python scripts/train.py
```


## Contact

For questions, please contact: [xfezhng@gmail.com]















