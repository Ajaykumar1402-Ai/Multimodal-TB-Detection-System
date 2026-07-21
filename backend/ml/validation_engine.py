import cv2
import numpy as np
from PIL import Image
import math

class ValidationError(Exception):
    pass

class ValidationEngine:
    def __init__(self):
        self.min_resolution = (200, 200)
        self.max_aspect_ratio = 2.5
        self.min_aspect_ratio = 0.4
        
        # Quality thresholds
        self.min_entropy = 4.0     # Below this is too uniform (blank/cartoon)
        self.min_laplacian = 10.0  # Below this is too blurry
        self.min_contrast = 20.0   # Difference between 5th and 95th percentile

    def validate_file_and_decode(self, image_path):
        """Task 3: File and Decode Validation"""
        try:
            # We open with cv2 to get the raw channels
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValidationError("File Validation Failed: Unable to decode image.")
            return img
        except Exception as e:
            raise ValidationError(f"File Validation Failed: {str(e)}")

    def validate_resolution_and_aspect_ratio(self, img):
        """Task 3: Resolution and Aspect Ratio Check"""
        h, w = img.shape[:2]
        if w < self.min_resolution[0] or h < self.min_resolution[1]:
            raise ValidationError(f"Resolution Check Failed: Image too small ({w}x{h}). Minimum is {self.min_resolution}.")
            
        aspect_ratio = w / float(h)
        if aspect_ratio > self.max_aspect_ratio or aspect_ratio < self.min_aspect_ratio:
            raise ValidationError(f"Aspect Ratio Check Failed: Unnatural aspect ratio ({aspect_ratio:.2f}).")

    def validate_rgb_consistency(self, img):
        """Task 4: RGB Consistency Check
        If R ≈ G ≈ B, convert to grayscale.
        Otherwise, reject as a natural color photograph.
        """
        if len(img.shape) < 3:
            return img # Already grayscale
            
        channels = cv2.split(img)
        if len(channels) >= 3:
            b, g, r = channels[:3]
            # Calculate mean absolute difference between channels
            diff_rg = np.mean(np.abs(r.astype(np.int16) - g.astype(np.int16)))
            diff_gb = np.mean(np.abs(g.astype(np.int16) - b.astype(np.int16)))
            diff_rb = np.mean(np.abs(r.astype(np.int16) - b.astype(np.int16)))
            
            avg_diff = (diff_rg + diff_gb + diff_rb) / 3.0
            
            if avg_diff > 15.0: # Threshold for color variation
                raise ValidationError(f"RGB Consistency Check Failed: Image contains natural colors (Color variance: {avg_diff:.2f}). True X-rays must be grayscale.")
            
            # Convert to grayscale and continue
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def validate_histogram_and_quality(self, gray_img):
        """Task 5 & 6: Histogram Analysis and Quality Check"""
        # Contrast check
        p5 = np.percentile(gray_img, 5)
        p95 = np.percentile(gray_img, 95)
        contrast = p95 - p5
        if contrast < self.min_contrast:
            raise ValidationError(f"Quality Check Failed: Extremely low contrast ({contrast:.2f}). Image may be blank or washed out.")
            
        # Check if completely black or white
        mean_val = np.mean(gray_img)
        if mean_val < 5:
            raise ValidationError("Quality Check Failed: Image is completely black.")
        if mean_val > 250:
            raise ValidationError("Quality Check Failed: Image is completely white.")

    def calculate_entropy(self, gray_img):
        """Helper to calculate Shannon entropy (Task 5)"""
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        logs = np.log2(hist + 1e-7)
        entropy = -np.sum(hist * logs)
        return entropy

    def validate_entropy_and_texture(self, gray_img):
        """Task 5 & 7: Entropy and Texture Analysis"""
        entropy = self.calculate_entropy(gray_img)
        if entropy < self.min_entropy:
            raise ValidationError(f"Entropy Analysis Failed: Image lacks detail (Entropy: {entropy:.2f}). Likely a synthetic or blank image.")
            
        laplacian_var = cv2.Laplacian(gray_img, cv2.CV_64F).var()
        if laplacian_var < self.min_laplacian:
            raise ValidationError(f"Texture Analysis Failed: Image is too blurry (Laplacian Variance: {laplacian_var:.2f}).")

    def verify_lung_presence(self, gray_img):
        """Task 8: Lung Presence Verification
        Uses a lightweight heuristic (adaptive thresholding + contour analysis)
        to check for large dark regions in the center corresponding to lungs.
        """
        # Resize for faster processing
        small = cv2.resize(gray_img, (256, 256))
        
        # Apply CLAHE to enhance structures
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enh = clahe.apply(small)
        
        # Threshold to find dark regions (lungs)
        _, thresh = cv2.threshold(enh, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        large_contours = [c for c in contours if cv2.contourArea(c) > (256 * 256 * 0.05)] # At least 5% of image
        
        if len(large_contours) < 1:
            raise ValidationError("Lung Verification Failed: Could not detect thoracic anatomy (lungs/rib cage).")
            
        # Optional: verify spatial distribution (left and right) if necessary.

    def run_pipeline(self, image_path):
        """Executes the entire multi-stage validation pipeline."""
        try:
            # 1 & 2
            img = self.validate_file_and_decode(image_path)
            self.validate_resolution_and_aspect_ratio(img)
            
            # 3 & 4
            gray_img = self.validate_rgb_consistency(img)
            
            # 5, 6, 7
            self.validate_histogram_and_quality(gray_img)
            self.validate_entropy_and_texture(gray_img)
            
            # 8
            self.verify_lung_presence(gray_img)
            
            # Return the processed grayscale image for the AI Guard Model
            return True, "Valid", gray_img
            
        except ValidationError as ve:
            return False, str(ve), None
        except Exception as e:
            return False, f"System Error during validation: {str(e)}", None

if __name__ == '__main__':
    engine = ValidationEngine()
    print("Validation Engine Loaded Successfully.")
