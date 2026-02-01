"""
Image Preprocessor for Document Processing

Handles image preprocessing tasks like deskewing, noise reduction,
contrast enhancement, and PDF to image conversion.
"""

import logging
from typing import List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from PIL import Image
import cv2
import io

try:
    from pdf2image import convert_from_path, convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing"""
    deskew: bool = True
    denoise: bool = True
    enhance_contrast: bool = True
    binarize: bool = False
    resize_max_dimension: Optional[int] = 2000
    dpi: int = 300


class ImagePreprocessor:
    """
    Preprocessor for document images.
    
    Provides:
    - PDF to image conversion
    - Deskewing
    - Noise reduction
    - Contrast enhancement
    - Image resizing
    - Region extraction
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
    
    def pdf_to_images(
        self,
        pdf_path: Union[str, Path, bytes],
        dpi: Optional[int] = None,
        first_page: Optional[int] = None,
        last_page: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Convert PDF to list of images.
        
        Args:
            pdf_path: Path to PDF file or PDF bytes
            dpi: Resolution for conversion
            first_page: First page to convert (1-indexed)
            last_page: Last page to convert (1-indexed)
            
        Returns:
            List of images as numpy arrays
        """
        if not PDF2IMAGE_AVAILABLE:
            raise ImportError("pdf2image is not installed. Run: pip install pdf2image")
        
        dpi = dpi or self.config.dpi
        
        try:
            if isinstance(pdf_path, bytes):
                pil_images = convert_from_bytes(
                    pdf_path,
                    dpi=dpi,
                    first_page=first_page,
                    last_page=last_page
                )
            else:
                pil_images = convert_from_path(
                    str(pdf_path),
                    dpi=dpi,
                    first_page=first_page,
                    last_page=last_page
                )
            
            # Convert PIL images to numpy arrays (BGR for OpenCV)
            images = []
            for pil_img in pil_images:
                img_array = np.array(pil_img)
                # Convert RGB to BGR for OpenCV
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                images.append(img_array)
            
            logger.info(f"Converted PDF to {len(images)} images at {dpi} DPI")
            return images
            
        except Exception as e:
            logger.error(f"Failed to convert PDF: {e}")
            raise
    
    def preprocess(
        self,
        image: np.ndarray,
        config: Optional[PreprocessingConfig] = None
    ) -> np.ndarray:
        """
        Apply full preprocessing pipeline to an image.
        
        Args:
            image: Input image as numpy array
            config: Optional override configuration
            
        Returns:
            Preprocessed image
        """
        cfg = config or self.config
        result = image.copy()
        
        # Resize if needed
        if cfg.resize_max_dimension:
            result = self.resize(result, cfg.resize_max_dimension)
        
        # Convert to grayscale for some operations
        if len(result.shape) == 3:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        else:
            gray = result.copy()
        
        # Deskew
        if cfg.deskew:
            result = self.deskew(result)
        
        # Denoise
        if cfg.denoise:
            result = self.denoise(result)
        
        # Enhance contrast
        if cfg.enhance_contrast:
            result = self.enhance_contrast(result)
        
        # Binarize
        if cfg.binarize:
            result = self.binarize(result)
        
        return result
    
    def resize(
        self,
        image: np.ndarray,
        max_dimension: int
    ) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: Input image
            max_dimension: Maximum width or height
            
        Returns:
            Resized image
        """
        height, width = image.shape[:2]
        
        if max(height, width) <= max_dimension:
            return image
        
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        
        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )
        
        return resized
    
    def deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Correct skew in document image.
        
        Uses Hough transform to detect text line angles.
        
        Args:
            image: Input image
            
        Returns:
            Deskewed image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )
        
        if lines is None or len(lines) == 0:
            return image
        
        # Calculate angles of detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Only consider near-horizontal lines
            if abs(angle) < 30:
                angles.append(angle)
        
        if not angles:
            return image
        
        # Use median angle to avoid outliers
        median_angle = np.median(angles)
        
        # Don't rotate if angle is very small
        if abs(median_angle) < 0.5:
            return image
        
        # Rotate image
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        
        # Calculate new image bounds
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        # Adjust rotation matrix
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Apply rotation with white background
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_width, new_height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255) if len(image.shape) == 3 else 255
        )
        
        logger.debug(f"Deskewed image by {median_angle:.2f} degrees")
        return rotated
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise from image.
        
        Uses Non-local Means Denoising.
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        if len(image.shape) == 3:
            # Color image
            denoised = cv2.fastNlMeansDenoisingColored(
                image,
                None,
                h=10,
                hColor=10,
                templateWindowSize=7,
                searchWindowSize=21
            )
        else:
            # Grayscale
            denoised = cv2.fastNlMeansDenoising(
                image,
                None,
                h=10,
                templateWindowSize=7,
                searchWindowSize=21
            )
        
        return denoised
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE.
        
        Args:
            image: Input image
            
        Returns:
            Contrast-enhanced image
        """
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge and convert back
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def binarize(
        self,
        image: np.ndarray,
        method: str = "otsu"
    ) -> np.ndarray:
        """
        Convert image to binary (black and white).
        
        Args:
            image: Input image
            method: Binarization method ('otsu', 'adaptive', 'simple')
            
        Returns:
            Binary image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if method == "otsu":
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif method == "adaptive":
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
        else:  # simple
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        return binary
    
    def extract_region(
        self,
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        padding: int = 0
    ) -> np.ndarray:
        """
        Extract a region from an image.
        
        Args:
            image: Source image
            x1, y1: Top-left corner
            x2, y2: Bottom-right corner
            padding: Extra pixels to include around the region
            
        Returns:
            Extracted region
        """
        height, width = image.shape[:2]
        
        # Apply padding with bounds checking
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        
        return image[y1:y2, x1:x2].copy()
    
    def to_pil(self, image: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image"""
        if len(image.shape) == 3:
            # BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image_rgb)
        else:
            return Image.fromarray(image)
    
    def from_pil(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to numpy array"""
        img_array = np.array(pil_image)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # RGB to BGR
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return img_array
    
    def to_bytes(
        self,
        image: np.ndarray,
        format: str = "PNG"
    ) -> bytes:
        """Convert image to bytes"""
        pil_image = self.to_pil(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        return buffer.getvalue()
    
    def from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Load image from bytes"""
        pil_image = Image.open(io.BytesIO(image_bytes))
        return self.from_pil(pil_image)
    
    def get_image_info(self, image: np.ndarray) -> dict:
        """Get image metadata"""
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        
        return {
            "width": width,
            "height": height,
            "channels": channels,
            "dtype": str(image.dtype),
            "size_bytes": image.nbytes
        }
