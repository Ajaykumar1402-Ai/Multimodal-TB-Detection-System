import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

class TemperatureScaler(nn.Module):
    """Wraps single-logit outputs with a learnable temperature."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        # logits: [N] or [N,1], temperature: scalar
        return logits / self.temperature

def calibrate():
    print("Calibrating model using Temperature Scaling...")
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    logits_path = os.path.join(models_dir, 'val_logits.json')
    
    if not os.path.exists(logits_path):
        print(f"Error: {logits_path} not found. Run optimize_threshold.py first.")
        return
        
    with open(logits_path, 'r') as f:
        data = json.load(f)
    
    # logits are scalars (single-logit binary model)
    raw_logits = data['logits']
    # Each entry may be a list [val] or a scalar float
    if isinstance(raw_logits[0], list):
        raw_logits = [x[0] for x in raw_logits]
    
    logits = torch.tensor(raw_logits, dtype=torch.float32)  # [N]
    labels = torch.tensor(data['labels'], dtype=torch.float32)  # [N]
    
    # Binary cross-entropy with logits (correct for single-logit model)
    bce = nn.BCEWithLogitsLoss()
    
    ece_before = bce(logits, labels).item()
    print(f"Before Temperature - BCE Loss: {ece_before:.4f}")
    
    model = TemperatureScaler()
    optimizer = optim.LBFGS([model.temperature], lr=0.01, max_iter=50)
    
    def eval_step():
        optimizer.zero_grad()
        loss = bce(model(logits), labels)
        loss.backward()
        return loss
        
    optimizer.step(eval_step)
    
    optimal_temperature = model.temperature.item()
    ece_after = bce(model(logits), labels).item()
    
    print(f"Optimal Temperature: {optimal_temperature:.4f}")
    print(f"After Temperature  - BCE Loss: {ece_after:.4f}")
    
    config_path = os.path.join(models_dir, 'densenet_config.json')
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            
    config['temperature'] = optimal_temperature
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"Saved temperature {optimal_temperature:.4f} to config.")

if __name__ == '__main__':
    calibrate()
