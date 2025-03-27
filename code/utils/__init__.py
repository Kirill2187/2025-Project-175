from .config import load_config
from .dataset import create_datasets, mnist_collate_fn
from .model import create_model
from .adversarial import AdversarialOptimizer

import logging
import numpy as np
import torch

def setup_logging(log_level):
    logging.basicConfig(level=log_level, format='%(levelname)s - %(message)s')
    
    
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    