import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import models, transforms
from PIL import Image

class GuardGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        
        # We only have one class output if we are using BCEWithLogitsLoss
        # If it's a 1D tensor representing logits of class 1
        loss = output[0]
        loss.backward()
        
        # Global average pooling on gradients
        weights = torch.mean(self.gradients, dim=(2, 3))[0]
        
        # Weight the activations
        activations = self.activations[0]
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = F.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)
        return cam.detach().cpu().numpy()

def analyze_guard_confidence(guard_model, image_path):
    """
    Task 6: Confidence-Based Decision & GradCAM Explainability
    Outputs: Valid Chest X-ray, Probably Chest X-ray, Uncertain, Probably Non-CXR, Invalid
    """
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img).unsqueeze(0)
    device = next(guard_model.parameters()).device
    tensor = tensor.to(device)
    
    # Run Inference
    with torch.no_grad():
        output = guard_model(tensor).squeeze()
        prob = torch.sigmoid(output).item()
        
    # Updated Thresholds based on deep training optimization (AUC=1.0)
    # The optimal decision boundary sits around 0.44
    if prob >= 0.44:
        confidence = "Valid Chest X-ray"
    elif prob >= 0.30:
        confidence = "Probably Chest X-ray"
    elif prob >= 0.15:
        confidence = "Uncertain"
    elif prob >= 0.05:
        confidence = "Probably Non-CXR"
    else:
        confidence = "Invalid"
        
    # Generate GradCAM if valid
    if prob >= 0.5:
        try:
            # Re-enable gradients for GradCAM
            tensor.requires_grad = True
            # MobileNetV2 last conv layer: model.features[-1]
            cam_extractor = GuardGradCAM(guard_model, guard_model.features[-1])
            cam = cam_extractor.generate(tensor)
            
            # Upsample CAM to 224x224
            cam_resized = cv2.resize(cam, (224, 224))
            
            # Basic anatomical verification:
            # Lungs should be in the center/middle regions, not strictly at the extreme borders.
            # We can check if the center of mass of the activation is roughly in the center.
            center_mass = np.mean(cam_resized[56:168, 56:168])
            edge_mass = (np.mean(cam_resized[:20, :]) + np.mean(cam_resized[-20:, :])) / 2
            
            if center_mass < edge_mass:
                # Attention is focused on the borders (possibly reading text/labels instead of lungs)
                # Lower the confidence
                confidence = "Uncertain (Anatomical Focus Warning)"
        except Exception as e:
            print(f"GradCAM Explainability skipped: {e}")
            
    return confidence, prob
