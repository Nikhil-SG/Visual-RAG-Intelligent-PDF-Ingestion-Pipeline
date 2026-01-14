"""
Rule-Based Image Classification Module

This module implements a computer vision-based classifier to categorize images extracted
from PDFs. It avoids deep learning overhead by using classical feature extraction
(color histograms, edge density, texture analysis, etc.) and comparing unknown images
against a set of manually curated reference images.

Key Features:
    - Dynamic Category Discovery: categories are inferred from reference folder structure.
    - Feature Extraction: computes geometric, color, and texture metrics.
    - Weighted Similarity: compares feature vectors to classify images as 'Content', 'Logo', etc.
"""

import os
import re
from PIL import Image, ImageStat
import numpy as np
import cv2
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_categories_from_reference_folder(reference_folder, default_min=65, default_max=100):
    """Dynamically discover categories from subfolders in reference folder"""
    category_thresholds = {}
    
    if not os.path.exists(reference_folder):
        logger.warning(f"Reference folder not found: {reference_folder}")
        return category_thresholds
    
    for item in os.listdir(reference_folder):
        item_path = os.path.join(reference_folder, item)
        if os.path.isdir(item_path):
            category_thresholds[item] = {'min': default_min, 'max': default_max}
            logger.info(f"Discovered category: {item}")
    
    return category_thresholds


class AdvancedImageClassifier:
    def __init__(self, reference_folder, category_thresholds=None):
        """Initialize advanced image classifier with custom thresholds"""
        self.reference_folder = reference_folder
        self.reference_features = {}
        
        # Dynamically get categories from reference folder if not provided
        if category_thresholds is None:
            self.category_thresholds = get_categories_from_reference_folder(reference_folder)
        else:
            self.category_thresholds = category_thresholds
            
        self.dynamic_thresholds = {}
        self.category_stats = {}
        
        logger.info("Loading reference image features...")
        self._load_reference_features()
        self._calculate_dynamic_thresholds()

    def _extract_enhanced_features(self, image_path):
        """Extract comprehensive features for better similarity detection"""
        try:
            img = Image.open(image_path).convert('RGB')
            img_cv = cv2.imread(image_path)
            
            if img_cv is None:
                logger.error(f"Could not read image: {image_path}")
                return None
            
            features = {}
            
            # Size features
            width, height = img.size
            features['width'] = width
            features['height'] = height
            features['aspect_ratio'] = width / height if height > 0 else 1.0
            features['area'] = width * height
            features['log_area'] = np.log10(max(width * height, 1))
            
            # Color statistics
            stat = ImageStat.Stat(img)
            features['mean_rgb'] = np.array(stat.mean)
            features['std_rgb'] = np.array(stat.stddev)
            features['brightness'] = np.mean(stat.mean)
            features['contrast'] = np.mean(stat.stddev)
            
            # Edge detection
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            edges_low = cv2.Canny(gray, 30, 100)
            edges_high = cv2.Canny(gray, 100, 200)
            features['edge_density_low'] = np.sum(edges_low > 0) / (width * height)
            features['edge_density_high'] = np.sum(edges_high > 0) / (width * height)
            
            # Texture
            features['texture_variance'] = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Binary threshold
            _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            features['binary_ratio'] = np.sum(binary_otsu == 255) / (width * height)
            
            # Color histograms
            hist_r = cv2.calcHist([img_cv], [2], None, [64], [0, 256])
            hist_g = cv2.calcHist([img_cv], [1], None, [64], [0, 256])
            hist_b = cv2.calcHist([img_cv], [0], None, [64], [0, 256])
            features['hist_r'] = cv2.normalize(hist_r, hist_r).flatten()
            features['hist_g'] = cv2.normalize(hist_g, hist_g).flatten()
            features['hist_b'] = cv2.normalize(hist_b, hist_b).flatten()
            
            # HSV features
            img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            hsv_mean = np.mean(img_hsv.reshape(-1, 3), axis=0)
            hsv_std = np.std(img_hsv.reshape(-1, 3), axis=0)
            features['hue_mean'] = hsv_mean[0]
            features['sat_mean'] = hsv_mean[1]
            features['val_mean'] = hsv_mean[2]
            features['hue_std'] = hsv_std[0]
            features['sat_std'] = hsv_std[1]
            features['val_std'] = hsv_std[2]
            
            # Corner detection
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
            features['corner_count'] = len(corners) if corners is not None else 0
            features['corner_density'] = features['corner_count'] / (width * height) * 10000
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {e}")
            return None

    def _calculate_advanced_similarity(self, features1, features2):
        """Enhanced similarity calculation with weighted feature importance"""
        if not features1 or not features2:
            return 0.0
        
        similarities = []
        weights = []
        
        # Geometry similarity
        if 'aspect_ratio' in features1 and 'aspect_ratio' in features2:
            ratio_diff = abs(features1['aspect_ratio'] - features2['aspect_ratio'])
            max_ratio = max(features1['aspect_ratio'], features2['aspect_ratio'])
            geom_sim = 1 - min(ratio_diff / max_ratio, 1.0)
            similarities.append(geom_sim)
            weights.append(2.0)
        
        # Area similarity
        if 'log_area' in features1 and 'log_area' in features2:
            area_diff = abs(features1['log_area'] - features2['log_area'])
            area_sim = max(0, 1 - area_diff / 2)
            similarities.append(area_sim)
            weights.append(1.5)
        
        # Color similarity
        if 'mean_rgb' in features1 and 'mean_rgb' in features2:
            color_diff = np.linalg.norm(features1['mean_rgb'] - features2['mean_rgb']) / 255.0
            color_sim = max(0, 1 - color_diff / np.sqrt(3))
            similarities.append(color_sim)
            weights.append(2.5)
        
        # Brightness similarity
        if 'brightness' in features1 and 'brightness' in features2:
            bright_diff = abs(features1['brightness'] - features2['brightness']) / 255.0
            bright_sim = 1 - bright_diff
            similarities.append(bright_sim)
            weights.append(1.8)
        
        # Edge similarity
        edge_keys = ['edge_density_low', 'edge_density_high']
        for key in edge_keys:
            if key in features1 and key in features2:
                edge_diff = abs(features1[key] - features2[key])
                edge_sim = 1 - min(edge_diff * 2, 1.0)
                similarities.append(edge_sim)
                weights.append(3.0)
        
        # Texture similarity
        if 'texture_variance' in features1 and 'texture_variance' in features2:
            max_var = max(features1['texture_variance'], features2['texture_variance'], 1e-6)
            var_diff = abs(features1['texture_variance'] - features2['texture_variance'])
            texture_sim = 1 - min(var_diff / max_var, 1.0)
            similarities.append(texture_sim)
            weights.append(1.5)
        
        # Binary ratio similarity
        if 'binary_ratio' in features1 and 'binary_ratio' in features2:
            binary_diff = abs(features1['binary_ratio'] - features2['binary_ratio'])
            binary_sim = 1 - binary_diff
            similarities.append(binary_sim)
            weights.append(2.0)
        
        # HSV similarity
        hsv_keys = ['hue_mean', 'sat_mean', 'val_mean']
        for key in hsv_keys:
            if key in features1 and key in features2:
                if key == 'hue_mean':
                    hue_diff = min(abs(features1[key] - features2[key]), 
                                 360 - abs(features1[key] - features2[key]))
                    hsv_sim = 1 - hue_diff / 180.0
                else:
                    hsv_diff = abs(features1[key] - features2[key]) / 255.0
                    hsv_sim = 1 - hsv_diff
                similarities.append(hsv_sim)
                weights.append(1.5)
        
        # Histogram similarity
        hist_keys = ['hist_r', 'hist_g', 'hist_b']
        for key in hist_keys:
            if key in features1 and key in features2:
                corr = cv2.compareHist(features1[key].astype(np.float32), 
                                      features2[key].astype(np.float32), 
                                      cv2.HISTCMP_CORREL)
                hist_sim = max(0, corr)
                similarities.append(hist_sim)
                weights.append(2.0)
        
        # Corner similarity
        if 'corner_density' in features1 and 'corner_density' in features2:
            max_corner = max(features1['corner_density'], features2['corner_density'], 1e-6)
            corner_diff = abs(features1['corner_density'] - features2['corner_density'])
            corner_sim = 1 - min(corner_diff / max_corner, 1.0)
            similarities.append(corner_sim)
            weights.append(2.5)
        
        if similarities and weights:
            weighted_sim = np.average(similarities, weights=weights)
            return weighted_sim
        
        return 0.0

    def _load_reference_features(self):
        """Load reference features with enhanced feature extraction"""
        categories = list(self.category_thresholds.keys())
        
        for category in categories:
            category_path = os.path.join(self.reference_folder, category)
            
            if not os.path.exists(category_path):
                logger.warning(f"Reference category folder not found: {category_path}")
                continue
            
            features_list = []
            image_files = [f for f in os.listdir(category_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            
            for img_file in image_files:
                img_path = os.path.join(category_path, img_file)
                features = self._extract_enhanced_features(img_path)
                if features is not None:
                    features_list.append(features)
            
            if features_list:
                self.reference_features[category] = features_list
                logger.info(f"Loaded {len(features_list)} reference images for {category}")
            else:
                logger.warning(f"No valid reference images found for {category}")

    def _is_monochrome(self, image_path, saturation_threshold=15, std_threshold=10):
        """Detect if an image is monochrome (single color)"""
        try:
            img = Image.open(image_path).convert('RGB')
            img_cv = cv2.imread(image_path)
            
            if img_cv is None:
                return False
            
            # Convert to HSV for saturation analysis
            img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            saturation_mean = np.mean(img_hsv[:, :, 1])
            
            # Check RGB standard deviation
            stat = ImageStat.Stat(img)
            rgb_std = np.mean(stat.stddev)
            
            # Image is monochrome if saturation is very low OR color variance is very low
            is_mono = saturation_mean < saturation_threshold or rgb_std < std_threshold
            
            if is_mono:
                logger.debug(f"Monochrome detected: {image_path} (sat={saturation_mean:.1f}, std={rgb_std:.1f})")
            
            return is_mono
            
        except Exception as e:
            logger.error(f"Error checking monochrome for {image_path}: {e}")
            return False

    def _calculate_dynamic_thresholds(self):
        """Calculate thresholds based on custom min/max settings"""
        for category, features_list in self.reference_features.items():
            min_threshold = self.category_thresholds[category]['min'] / 100.0
            max_threshold = self.category_thresholds[category]['max'] / 100.0
            
            if len(features_list) < 2:
                self.dynamic_thresholds[category] = min_threshold
                continue
            
            similarities = []
            for i in range(len(features_list)):
                for j in range(i + 1, len(features_list)):
                    sim = self._calculate_advanced_similarity(features_list[i], features_list[j])
                    similarities.append(sim)
            
            if similarities:
                mean_sim = np.mean(similarities)
                std_sim = np.std(similarities)
                min_sim = np.min(similarities)
                max_sim = np.max(similarities)
                q25 = np.percentile(similarities, 25)
                q75 = np.percentile(similarities, 75)
                
                self.category_stats[category] = {
                    'mean': mean_sim,
                    'std': std_sim,
                    'min': min_sim,
                    'max': max_sim,
                    'q25': q25,
                    'q75': q75,
                    'count': len(similarities)
                }
                
                consistency = 1 - std_sim
                
                if consistency > 0.9:
                    threshold = max(min_threshold, mean_sim - 0.05)
                elif consistency > 0.8:
                    threshold = max(min_threshold, mean_sim - 0.1)
                elif consistency > 0.6:
                    threshold = max(min_threshold, mean_sim - 0.15)
                else:
                    threshold = max(min_threshold, q25)
                
                threshold = min(threshold, max_threshold)
                self.dynamic_thresholds[category] = threshold
                
                logger.info(f"{category}: mean={mean_sim:.3f}, std={std_sim:.3f}, "
                          f"consistency={consistency:.3f}, threshold={threshold:.3f} "
                          f"(range: {min_threshold}-{max_threshold})")
            else:
                self.dynamic_thresholds[category] = min_threshold

    def classify_image(self, image_path):
        """Enhanced image classification with detailed scoring"""
        features = self._extract_enhanced_features(image_path)
        
        if features is None:
            return 'unknown', 0.0, {}, {}
        
        scores = {}
        detailed_scores = {}
        
        for category, ref_features_list in self.reference_features.items():
            category_similarities = []
            
            for ref_features in ref_features_list:
                sim = self._calculate_advanced_similarity(features, ref_features)
                category_similarities.append(sim)
            
            if category_similarities:
                max_sim = np.max(category_similarities)
                mean_sim = np.mean(category_similarities)
                scores[category] = max_sim
                
                detailed_scores[category] = {
                    'max_similarity': max_sim,
                    'mean_similarity': mean_sim,
                    'threshold': self.dynamic_thresholds.get(category, 0.85),
                    'meets_threshold': max_sim >= self.dynamic_thresholds.get(category, 0.85),
                    'reference_count': len(category_similarities)
                }
        
        if not scores:
            # No reference categories found, check if monochrome
            if self._is_monochrome(image_path):
                return 'monochrome', 1.0, {}, {}
            return 'unknown', 0.0, {}, {}
        
        valid_categories = []
        for category, score in scores.items():
            threshold = self.dynamic_thresholds.get(category, 0.85)
            if score >= threshold:
                valid_categories.append((category, score))
        
        if valid_categories:
            valid_categories.sort(key=lambda x: x[1], reverse=True)
            best_category, best_score = valid_categories[0]
            return best_category, best_score, scores, detailed_scores
        else:
            # No category meets threshold - check if image is monochrome
            if self._is_monochrome(image_path):
                return 'monochrome', 1.0, scores, detailed_scores
            
            best_category = max(scores.keys(), key=lambda k: scores[k])
            best_score = scores[best_category]
            
            if best_score < 0.3:
                return 'unknown', best_score, scores, detailed_scores
            else:
                # Return the best scoring category even if it doesn't meet threshold
                # This avoids hardcoding category names that might not exist
                return best_category, best_score, scores, detailed_scores

    def get_detailed_summary(self):
        """Get comprehensive summary of thresholds and statistics"""
        summary = "\n=== ADVANCED CLASSIFIER SUMMARY ===\n"
        
        for category in self.dynamic_thresholds:
            threshold = self.dynamic_thresholds[category]
            min_thresh = self.category_thresholds[category]['min']
            max_thresh = self.category_thresholds[category]['max']
            
            summary += f"\n{category.upper()}:\n"
            summary += f"  Threshold: {threshold:.3f} (range: {min_thresh}-{max_thresh}%)\n"
            
            if category in self.category_stats:
                stats = self.category_stats[category]
                summary += f"  Reference images: {len(self.reference_features[category])}\n"
                summary += f"  Similarity stats: mean={stats['mean']:.3f}, std={stats['std']:.3f}\n"
                summary += f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]\n"
                summary += f"  Quartiles: Q25={stats['q25']:.3f}, Q75={stats['q75']:.3f}\n"
        
        return summary


def classify_and_organize_images(image_folder, reference_folder, output_folder):
    """Main function to classify and organize images"""
    # Dynamically get categories from reference folder
    category_thresholds = get_categories_from_reference_folder(reference_folder)
    
    if not category_thresholds:
        logger.error("No categories found in reference folder!")
        return {}, {}
    
    logger.info(f"Found {len(category_thresholds)} categories: {list(category_thresholds.keys())}")
    
    classifier = AdvancedImageClassifier(reference_folder, category_thresholds)
    logger.info(classifier.get_detailed_summary())
    
    os.makedirs(output_folder, exist_ok=True)
    
    categories = list(category_thresholds.keys()) + ['unknown', 'monochrome']
    for category in categories:
        os.makedirs(os.path.join(output_folder, category), exist_ok=True)
    
    classification_results = {}
    organized_images = {category: [] for category in categories}
    
    # MODIFIED: Use os.walk to find images in subfolders recursively
    image_paths = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_paths.append(os.path.join(root, file))
    
    logger.info(f"Classifying {len(image_paths)} images...")
    
    for i, image_path in enumerate(image_paths):
        image_file = os.path.basename(image_path)
        
        category, score, all_scores, detailed_scores = classifier.classify_image(image_path)
        
        classification_results[image_file] = {
            'category': category,
            'score': score,
            'all_scores': all_scores,
            'detailed_scores': detailed_scores,
            'path': image_path,
            'threshold_used': classifier.dynamic_thresholds.get(category, 0.85)
        }
        
        category_folder = os.path.join(output_folder, category)
        dest_path = os.path.join(category_folder, image_file)
        
        # Handle duplicate filenames
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(image_file)
            dest_path = os.path.join(category_folder, f"{base}_{i}{ext}")
            
        shutil.copy2(image_path, dest_path)
        
        organized_images[category].append({
            'original_path': image_path,
            'organized_path': dest_path,
            'score': score,
            'all_scores': all_scores,
            'detailed_scores': detailed_scores
        })
        
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(image_paths)} images")
        
        threshold_used = classifier.dynamic_thresholds.get(category, 0.85)
        confidence = 'high' if score >= threshold_used else 'low'
        all_scores_str = ', '.join([f"{cat}: {sc:.2f}" for cat, sc in all_scores.items()])
        
        logger.info(f"{image_file} -> {category} (score: {score:.2f}, "
                   f"threshold: {threshold_used:.2f}, conf: {confidence}) "
                   f"[{all_scores_str}]")
    
    logger.info("\n=== CLASSIFICATION RESULTS SUMMARY ===")
    total_images = 0
    for category, images in organized_images.items():
        scores = [img['score'] for img in images]
        if scores:
            avg_score = np.mean(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            total_images += len(images)
            logger.info(f"{category}: {len(images)} images "
                       f"(avg: {avg_score:.2f}, range: {min_score:.2f}-{max_score:.2f})")
        else:
            logger.info(f"{category}: 0 images")
    
    logger.info(f"\nTotal images processed: {total_images}")
    logger.info(f"Images organized in: {output_folder}")
    
    return classification_results, organized_images


if __name__ == "__main__":
    IMAGE_FOLDER = r"E:\Master's\2nd Year\MyWork\Data\Images"  # Your input images folder
    REFERENCE_FOLDER = r"E:\Master's\2nd Year\MyWork\Data\Reference_Images"  # Your reference folder with animal subfolders
    OUTPUT_FOLDER = r"E:\Master's\2nd Year\MyWork\Data\classified_images"  # Output folder for organized images
    
    print("=== IMAGE SIMILARITY CLASSIFIER ===")
    
    results, organized = classify_and_organize_images(
        IMAGE_FOLDER,
        REFERENCE_FOLDER,
        OUTPUT_FOLDER
    )
    
    print(f"\nClassification complete! Check '{OUTPUT_FOLDER}' for organized images.")