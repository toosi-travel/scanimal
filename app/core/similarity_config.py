"""
Configuration for enhanced similarity calculation and color discrimination
"""

class SimilarityConfig:
    """Configuration for similarity calculation settings"""
    
    # Color sensitivity settings
    COLOR_WEIGHT = 0.6                    # Weight for color features in embeddings
    STRUCTURE_WEIGHT = 0.4                # Weight for structural features in embeddings
    
    # Color similarity thresholds for pre-filtering
    COLOR_SIMILARITY_LOW = 0.3           # Below this: apply 0.3 penalty
    COLOR_SIMILARITY_MEDIUM = 0.6        # Below this: apply 0.7 penalty
    COLOR_SIMILARITY_HIGH = 1.0          # Above this: no penalty
    
    # Color penalties
    COLOR_PENALTY_LOW = 0.3              # Penalty for very low color similarity
    COLOR_PENALTY_MEDIUM = 0.7           # Penalty for medium color similarity
    COLOR_PENALTY_HIGH = 1.0             # No penalty for high color similarity
    
    # Enhanced similarity metric weights
    COSINE_WEIGHT = 0.5                  # Weight for cosine similarity
    EUCLIDEAN_WEIGHT = 0.3               # Weight for Euclidean distance
    MANHATTAN_WEIGHT = 0.2               # Weight for Manhattan distance
    
    # Distance normalization factors
    EUCLIDEAN_SCALE_FACTOR = 1.0         # Scale factor for Euclidean distance
    MANHATTAN_SCALE_FACTOR = 100.0       # Scale factor for Manhattan distance
    
    # Duplicate detection thresholds (much stricter for better color discrimination)
    AUTO_REJECT_THRESHOLD = 0.85         # Below this: auto-reject
    PENDING_THRESHOLD_MIN = 0.90         # Below this: pending approval
    AUTO_REGISTER_THRESHOLD = 0.95       # Above this: auto-register
    
    # Feature extraction settings
    COLOR_HISTOGRAM_BINS = {
        'rgb': 64,                       # RGB histogram bins
        'hsv': 32,                       # HSV histogram bins
        'lab': 32,                       # LAB histogram bins
    }
    
    DOMINANT_COLORS_K = 8                # Number of dominant colors to extract
    TEXTURE_FEATURE_SIZE = 50            # Size of texture feature vector
    SHAPE_FEATURE_SIZE = 10              # Size of shape feature vector
    LBP_FEATURE_SIZE = 10                # Size of LBP feature vector
    
    # Image preprocessing
    COLOR_COMPARISON_SIZE = (64, 64)     # Size for color similarity comparison
    EMBEDDING_TARGET_SIZE = (112, 112)   # Target size for ArcFace embeddings
    
    # Gabor filter parameters
    GABOR_ANGLES = [0, 45, 90, 135]     # Gabor filter angles
    GABOR_FREQUENCIES = [0.1, 0.3, 0.5] # Gabor filter frequencies
    GABOR_KERNEL_SIZE = (21, 21)         # Gabor kernel size
    
    # LBP parameters
    LBP_POINTS_8 = 8                     # LBP points for 8-neighborhood
    LBP_POINTS_16 = 16                   # LBP points for 16-neighborhood
    LBP_RADIUS_1 = 1                     # LBP radius for first scale
    LBP_RADIUS_2 = 2                     # LBP radius for second scale
    
    # PCA settings
    PCA_TARGET_DIMENSIONS = 512          # Target dimensions after PCA reduction
    
    # Logging settings
    LOG_COLOR_SIMILARITY = True          # Log color similarity scores
    LOG_ENHANCED_SIMILARITY = True       # Log enhanced similarity breakdown
    LOG_FEATURE_EXTRACTION = False       # Log feature extraction details (verbose)

class Thresholds:
    """Legacy thresholds class for backward compatibility"""
    
    def __init__(self):
        self.auto_reject_threshold = SimilarityConfig.AUTO_REJECT_THRESHOLD
        self.pending_threshold_min = SimilarityConfig.PENDING_THRESHOLD_MIN
        self.auto_register_threshold = SimilarityConfig.AUTO_REGISTER_THRESHOLD

# Global configuration instance
similarity_config = SimilarityConfig() 