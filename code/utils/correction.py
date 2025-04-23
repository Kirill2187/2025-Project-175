import numpy as np
import logging
import torch
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import scipy.stats as sps

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
        
class HardGMMCorrectionStrategy(BaseCorrectionStrategy):
    def __init__(self, model, train_dataset, threshold=0.95):
        self.model = model
        self.train_dataset = train_dataset
        self.threshold = threshold
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.intersections = []
        self.prev_corruption_probas = None
        
    @staticmethod
    def _gaussian_intersection(mu1, sigma1, mu2, sigma2):
        pdf1 = sps.norm(mu1, sigma1).pdf
        pdf2 = sps.norm(mu2, sigma2).pdf
        
        def integrand(x):
            return np.minimum(pdf1(x), pdf2(x))
        
        lower_bound = min(mu1 - 5 * sigma1, mu2 - 5 * sigma2)
        upper_bound = max(mu1 + 5 * sigma1, mu2 + 5 * sigma2)
        
        from scipy import integrate
        return integrate.quad(integrand, lower_bound, upper_bound)[0]
    
    def step(self, losses: np.ndarray, *args, **kwargs):
        logging.debug("GMM correction step")
        
        losses = losses.copy()
        losses = (losses - np.mean(losses)) / np.std(losses)
        losses_no_outliers = losses[losses < np.quantile(losses, 0.975)]
                    
        gmm = GaussianMixture(n_components=2, covariance_type='diag')
        gmm.fit(losses_no_outliers.reshape(-1, 1))
        mu1, sigma1 = gmm.means_[0][0], np.sqrt(gmm.covariances_[0][0])
        mu2, sigma2 = gmm.means_[1][0], np.sqrt(gmm.covariances_[1][0])
        flipped = False
        if mu1 > mu2:
            flipped = True
            mu1, sigma1, mu2, sigma2 = mu2, sigma2, mu1, sigma1
        probas = gmm.predict_proba(losses.reshape(-1, 1))
        if flipped:
            probas = 1 - probas
        probas = probas[:, 1]
        
        self.intersections.append(self._gaussian_intersection(mu1, sigma1, mu2, sigma2))
        logging.debug(f"Fitted GMM: mu1={mu1}, sigma1={sigma1}, mu2={mu2}, sigma2={sigma2}")
        logging.info(f"Intersection: {self.intersections[-1]}")
        
        if len(self.intersections) == 1 or self.intersections[-1] < self.intersections[-2]:
            self.prev_corruption_probas = probas
            return
        
        # self.model.eval()
        # with torch.no_grad():
        #     predictions = np.zeros(len(losses), dtype=int)
        #     # losses = np.zeros(len(losses), dtype=float)
        #     for batch in tqdm(self.train_loader, desc="Evaluating losses"):
        #         images = batch['image']
        #         labels = batch['label']
        #         outputs = self.model(images)
        #         # batch_losses = self.criterion(outputs, labels).cpu().numpy()
        #         predictions[batch['index']] = outputs.argmax(dim=1).cpu().numpy()
        #         # losses[batch['index']] = batch_losses
                
        # self.model.train()
        
        predictions = kwargs['predictions']
        
        logging.debug("Intersection increased, applying correction")
            
        indices_to_correct = np.where(self.prev_corruption_probas >= self.threshold)[0]
        logging.debug(f"Correcting for {len(indices_to_correct)} samples")
        self.train_dataset.change_labels(indices_to_correct, predictions[indices_to_correct])
        
        logging.debug("Resetting model parameters")
        self.model.apply(lambda x: x.reset_parameters() if hasattr(x, 'reset_parameters') else None)
        
        self.intersections = []
        self.prev_corruption_probas = None
        

def create_correction_strategy(config, model, train_dataset):
    strategy_config = config['training'].get('correction', {})
    if not strategy_config:
        logging.info("No correction strategy specified, using default")
        return BaseCorrectionStrategy()
    strategy_type = strategy_config['method'].lower()
    
    if strategy_type == 'model':
        logging.info("Using model prediction correction strategy")
        return ModelPredictionCorrectionStrategy(model, train_dataset, **strategy_config['params'])
    elif strategy_type == 'gmm_hard':
        logging.info("Using GMM correction strategy")
        return HardGMMCorrectionStrategy(model, train_dataset, **strategy_config['params'])