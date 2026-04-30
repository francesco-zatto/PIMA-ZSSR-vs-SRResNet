from pathlib import Path

# Base paths - Adjust ROOT to your local path if not on Colab
ROOT = Path('/Users/linneaandersen/PIMA/PIMA-ZSSR-vs-SRResNet') 
URBAN100_DIR = ROOT / 'datasets/Urban100'

# Define your models and their specific architectural flags
model_definitions_resnet = {
    'model_allOFF': {
        'short_name': 'no BN',
        'bn': False, 'scaling': False, 'tanh': False
    },
    'model_bnON_scalingOFF_tanhOFF': {
        'short_name': 'paper architecture',
        'bn': True, 'scaling': False, 'tanh': False
    },
    'model_bnOFF_scalingON_tanhOFF': {
        'short_name': 'LR scaling',
        'bn': False, 'scaling': True, 'tanh': False
    }
}

# Build the dynamic dictionary
models = {}
for m_id, cfg in model_definitions_resnet.items():
    # Adjust this path to where your .pth files actually live
    checkpoint_dir = ROOT / 'checkpoints' / m_id
    
    models[m_id] = {
        'short_name': cfg['short_name'],
        'model_type': 'SRResNet',
        'checkpoint': checkpoint_dir / 'srresnet_final.pth',
        'use_batch_norm': cfg['bn'],
        'scale_lr': cfg['scaling'],
        'final_activation': cfg['tanh']
    }