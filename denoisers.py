import cv2
import numpy as np
from typing import Optional

class Denoiser:
    def __init__(self):
        self.available_methods = ['bilateral', 'nlmeans', 'gaussian', 'median']
    
    def denoise(self, image: np.ndarray, method: str = 'bilateral', **kwargs) -> np.ndarray:
        """
        Apply denoising to the image using specified method
        
        Args:
            image: Input image (0-1 range)
            method: Denoising method ('bilateral', 'nlmeans', 'gaussian', 'median')
            **kwargs: Method-specific parameters
        
        Returns:
            Denoised image
        """
        # Convert to 0-255 range for OpenCV
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        if method == 'bilateral':
            return self._bilateral_denoise(image_uint8, **kwargs)
        elif method == 'nlmeans':
            return self._nlmeans_denoise(image_uint8, **kwargs)
        elif method == 'gaussian':
            return self._gaussian_denoise(image_uint8, **kwargs)
        elif method == 'median':
            return self._median_denoise(image_uint8, **kwargs)
        else:
            raise ValueError(f"Unknown denoising method: {method}")
    
    def _bilateral_denoise(self, image: np.ndarray, 
                          d: int = 9, 
                          sigma_color: float = 75, 
                          sigma_space: float = 75) -> np.ndarray:
        """Bilateral filter denoising"""
        denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        return denoised.astype(np.float32) / 255.0
    
    def _nlmeans_denoise(self, image: np.ndarray,
                        h: float = 10,
                        template_window_size: int = 7,
                        search_window_size: int = 21) -> np.ndarray:
        """Non-local means denoising"""
        denoised = cv2.fastNlMeansDenoisingColored(
            image, 
            None, 
            h, h, 
            template_window_size, 
            search_window_size
        )
        return denoised.astype(np.float32) / 255.0
    
    def _gaussian_denoise(self, image: np.ndarray,
                         kernel_size: int = 5,
                         sigma: float = 1.0) -> np.ndarray:
        """Gaussian blur denoising"""
        denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        return denoised.astype(np.float32) / 255.0
    
    def _median_denoise(self, image: np.ndarray,
                       kernel_size: int = 5) -> np.ndarray:
        """Median filter denoising"""
        denoised = cv2.medianBlur(image, kernel_size)
        return denoised.astype(np.float32) / 255.0
    
    def compare_denoisers(self, image: np.ndarray) -> dict:
        """Compare all available denoising methods"""
        results = {}
        
        for method in self.available_methods:
            try:
                denoised = self.denoise(image, method)
                # Calculate metrics
                noise_reduction = self._calculate_noise_reduction(image, denoised)
                results[method] = {
                    'image': denoised,
                    'noise_reduction': noise_reduction
                }
            except Exception as e:
                print(f"Error with {method}: {e}")
                results[method] = None
        
        return results
    
    def _calculate_noise_reduction(self, original: np.ndarray, denoised: np.ndarray) -> float:
        """Calculate noise reduction percentage"""
        if len(original.shape) == 3:
            original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            denoised_gray = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)
        else:
            original_gray = original
            denoised_gray = denoised
        
        original_var = cv2.Laplacian(original_gray, cv2.CV_64F).var()
        denoised_var = cv2.Laplacian(denoised_gray, cv2.CV_64F).var()
        
        if original_var == 0:
            return 0.0
        
        return max(0, (1 - denoised_var / original_var)) * 100