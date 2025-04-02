import numpy as np
import logging
import torch

class BaseCorrectionStrategy:
    def __init__(self):
        pass
    
    def step(self, losses: np.ndarray, *args, **kwargs):
        pass
    

class ModelPredictionCorrectionStrategy(BaseCorrectionStrategy):
    def __init__(self, model, train_dataset, top_p=0.01, interval=5):
        super().__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.top_p = top_p
        self.interval = interval
        self.step_count = 0
    
    def step(self, losses: np.ndarray, *args, **kwargs):
        self.step_count += 1
        if self.step_count % self.interval != 0:
            return
        logging.debug("Model prediction correction step")
        
        self.model.eval()
        with torch.no_grad():
            threshold = np.percentile(losses, 100 * (1 - self.top_p))
            indices_to_correct = np.where(losses >= threshold)[0]
            logging.debug(f"Correcting for {len(indices_to_correct)} samples")
            
            predictions = np.zeros(len(losses), dtype=int)
            for index in indices_to_correct:
                image = self.train_dataset[index]['image'].unsqueeze(0)
                output = self.model(image)
                predictions[index] = output.argmax().item()

            self.train_dataset.change_labels(indices_to_correct, predictions[indices_to_correct])
            
        self.model.train()
        
        

def create_correction_strategy(config, model, train_dataset):
    strategy_config = config['training'].get('correction', {})
    if not strategy_config:
        logging.info("No correction strategy specified, using default")
        return BaseCorrectionStrategy()
    strategy_type = strategy_config['method'].lower()
    
    if strategy_type == 'model':
        logging.info("Using model prediction correction strategy")
        return ModelPredictionCorrectionStrategy(model, train_dataset, **strategy_config['params'])
    else:
        raise ValueError(f"Unknown correction strategy: {strategy_type}")