"""
Model registry for managing and versioning machine learning models.
"""

import pickle
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import hashlib
import shutil

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import joblib

from src.utils.logging_config import get_logger
from src.utils.error_handling import ModelError, ValidationError

logger = get_logger(__name__)


class ModelRegistry:
    """
    Registry for managing machine learning models with versioning and metadata.
    """
    
    def __init__(self, registry_path: str = "models/registry"):
        """
        Initialize the model registry.
        
        Args:
            registry_path: Path to the model registry directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.registry_path / "models").mkdir(exist_ok=True)
        (self.registry_path / "metadata").mkdir(exist_ok=True)
        (self.registry_path / "artifacts").mkdir(exist_ok=True)
        
        self.metadata_file = self.registry_path / "metadata" / "registry.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load registry metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """Save registry metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save registry metadata: {e}")
            raise ModelError(f"Failed to save registry metadata: {e}")
    
    def _generate_model_hash(self, model: BaseEstimator) -> str:
        """Generate a hash for the model."""
        try:
            # Serialize model to bytes
            model_bytes = pickle.dumps(model)
            # Generate hash
            return hashlib.md5(model_bytes).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to generate model hash: {e}")
            return hashlib.md5(str(datetime.now()).encode()).hexdigest()
    
    def register_model(self, 
                      model: BaseEstimator,
                      model_name: str,
                      model_type: str,
                      version: str = None,
                      description: str = "",
                      tags: List[str] = None,
                      metrics: Dict[str, float] = None,
                      hyperparameters: Dict[str, Any] = None,
                      training_data_info: Dict[str, Any] = None) -> str:
        """
        Register a new model in the registry.
        
        Args:
            model: The trained model
            model_name: Name of the model
            model_type: Type of model (e.g., 'clustering', 'regime_detection')
            version: Model version (auto-generated if None)
            description: Model description
            tags: List of tags
            metrics: Model performance metrics
            hyperparameters: Model hyperparameters
            training_data_info: Information about training data
            
        Returns:
            Model version string
        """
        try:
            # Generate version if not provided
            if version is None:
                version = self._generate_version(model_name)
            
            # Generate model hash
            model_hash = self._generate_model_hash(model)
            
            # Create model directory
            model_dir = self.registry_path / "models" / model_name / version
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = model_dir / "model.pkl"
            joblib.dump(model, model_path)
            
            # Create metadata
            model_metadata = {
                'model_name': model_name,
                'version': version,
                'model_type': model_type,
                'description': description,
                'tags': tags or [],
                'metrics': metrics or {},
                'hyperparameters': hyperparameters or {},
                'training_data_info': training_data_info or {},
                'model_hash': model_hash,
                'created_at': datetime.now().isoformat(),
                'model_path': str(model_path),
                'file_size_mb': model_path.stat().st_size / (1024 * 1024)
            }
            
            # Save metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(model_metadata, f, indent=2, default=str)
            
            # Update registry metadata
            if model_name not in self.metadata:
                self.metadata[model_name] = {}
            
            self.metadata[model_name][version] = model_metadata
            self._save_metadata()
            
            logger.info(f"Registered model {model_name} version {version}")
            return version
            
        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")
            raise ModelError(f"Model registration failed: {e}")
    
    def _generate_version(self, model_name: str) -> str:
        """Generate a new version number for a model."""
        if model_name in self.metadata:
            existing_versions = list(self.metadata[model_name].keys())
            if existing_versions:
                # Extract version numbers and find the highest
                version_numbers = []
                for version in existing_versions:
                    try:
                        # Handle semantic versioning (e.g., "1.0.0")
                        if '.' in version:
                            version_numbers.append([int(x) for x in version.split('.')])
                        else:
                            version_numbers.append([int(version)])
                    except ValueError:
                        continue
                
                if version_numbers:
                    # Increment the highest version
                    highest_version = max(version_numbers)
                    if len(highest_version) == 1:
                        return str(highest_version[0] + 1)
                    else:
                        highest_version[-1] += 1
                        return '.'.join(map(str, highest_version))
        
        return "1.0.0"
    
    def load_model(self, model_name: str, version: str = None) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """
        Load a model from the registry.
        
        Args:
            model_name: Name of the model
            version: Model version (loads latest if None)
            
        Returns:
            Tuple of (model, metadata)
        """
        try:
            if model_name not in self.metadata:
                raise ModelError(f"Model {model_name} not found in registry")
            
            # Get version
            if version is None:
                # Get latest version
                versions = list(self.metadata[model_name].keys())
                if not versions:
                    raise ModelError(f"No versions found for model {model_name}")
                version = max(versions, key=lambda v: self.metadata[model_name][v]['created_at'])
            
            if version not in self.metadata[model_name]:
                raise ModelError(f"Version {version} not found for model {model_name}")
            
            # Load model
            model_path = Path(self.metadata[model_name][version]['model_path'])
            if not model_path.exists():
                raise ModelError(f"Model file not found: {model_path}")
            
            model = joblib.load(model_path)
            metadata = self.metadata[model_name][version]
            
            logger.info(f"Loaded model {model_name} version {version}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name} version {version}: {e}")
            raise ModelError(f"Model loading failed: {e}")
    
    def list_models(self, model_type: str = None) -> List[Dict[str, Any]]:
        """
        List all models in the registry.
        
        Args:
            model_type: Filter by model type
            
        Returns:
            List of model information
        """
        models = []
        
        for model_name, versions in self.metadata.items():
            for version, metadata in versions.items():
                if model_type is None or metadata['model_type'] == model_type:
                    models.append({
                        'model_name': model_name,
                        'version': version,
                        'model_type': metadata['model_type'],
                        'description': metadata['description'],
                        'created_at': metadata['created_at'],
                        'metrics': metadata.get('metrics', {}),
                        'tags': metadata.get('tags', [])
                    })
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x['created_at'], reverse=True)
        return models
    
    def get_model_info(self, model_name: str, version: str = None) -> Dict[str, Any]:
        """
        Get detailed information about a model.
        
        Args:
            model_name: Name of the model
            version: Model version (gets latest if None)
            
        Returns:
            Model information dictionary
        """
        try:
            if model_name not in self.metadata:
                raise ModelError(f"Model {model_name} not found in registry")
            
            if version is None:
                # Get latest version
                versions = list(self.metadata[model_name].keys())
                if not versions:
                    raise ModelError(f"No versions found for model {model_name}")
                version = max(versions, key=lambda v: self.metadata[model_name][v]['created_at'])
            
            if version not in self.metadata[model_name]:
                raise ModelError(f"Version {version} not found for model {model_name}")
            
            return self.metadata[model_name][version]
            
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name} version {version}: {e}")
            raise ModelError(f"Failed to get model info: {e}")
    
    def delete_model(self, model_name: str, version: str) -> bool:
        """
        Delete a model from the registry.
        
        Args:
            model_name: Name of the model
            version: Model version
            
        Returns:
            True if successful
        """
        try:
            if model_name not in self.metadata or version not in self.metadata[model_name]:
                raise ModelError(f"Model {model_name} version {version} not found")
            
            # Delete model file
            model_path = Path(self.metadata[model_name][version]['model_path'])
            if model_path.exists():
                model_path.unlink()
            
            # Delete model directory
            model_dir = model_path.parent
            if model_dir.exists() and not any(model_dir.iterdir()):
                model_dir.rmdir()
            
            # Remove from metadata
            del self.metadata[model_name][version]
            
            # Remove model entry if no versions left
            if not self.metadata[model_name]:
                del self.metadata[model_name]
            
            self._save_metadata()
            
            logger.info(f"Deleted model {model_name} version {version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_name} version {version}: {e}")
            raise ModelError(f"Model deletion failed: {e}")
    
    def update_model_metrics(self, model_name: str, version: str, metrics: Dict[str, float]) -> bool:
        """
        Update model metrics.
        
        Args:
            model_name: Name of the model
            version: Model version
            metrics: New metrics
            
        Returns:
            True if successful
        """
        try:
            if model_name not in self.metadata or version not in self.metadata[model_name]:
                raise ModelError(f"Model {model_name} version {version} not found")
            
            # Update metrics
            self.metadata[model_name][version]['metrics'].update(metrics)
            self.metadata[model_name][version]['updated_at'] = datetime.now().isoformat()
            
            # Save updated metadata
            metadata_path = Path(self.metadata[model_name][version]['model_path']).parent / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata[model_name][version], f, indent=2, default=str)
            
            self._save_metadata()
            
            logger.info(f"Updated metrics for model {model_name} version {version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update metrics for model {model_name} version {version}: {e}")
            raise ModelError(f"Metrics update failed: {e}")
    
    def get_best_model(self, model_name: str, metric: str = None, higher_is_better: bool = True) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """
        Get the best performing model based on a metric.
        
        Args:
            model_name: Name of the model
            metric: Metric to optimize (uses first available if None)
            higher_is_better: Whether higher metric values are better
            
        Returns:
            Tuple of (model, metadata)
        """
        try:
            if model_name not in self.metadata:
                raise ModelError(f"Model {model_name} not found in registry")
            
            versions = list(self.metadata[model_name].keys())
            if not versions:
                raise ModelError(f"No versions found for model {model_name}")
            
            # Find best model
            best_version = None
            best_score = None
            
            for version in versions:
                metadata = self.metadata[model_name][version]
                metrics = metadata.get('metrics', {})
                
                if not metrics:
                    continue
                
                if metric is None:
                    # Use first available metric
                    metric = list(metrics.keys())[0]
                
                if metric in metrics:
                    score = metrics[metric]
                    if best_score is None or (higher_is_better and score > best_score) or (not higher_is_better and score < best_score):
                        best_score = score
                        best_version = version
            
            if best_version is None:
                raise ModelError(f"No models with metric {metric} found for {model_name}")
            
            return self.load_model(model_name, best_version)
            
        except Exception as e:
            logger.error(f"Failed to get best model for {model_name}: {e}")
            raise ModelError(f"Failed to get best model: {e}")
    
    def cleanup_old_models(self, days_to_keep: int = 30) -> int:
        """
        Clean up old model versions.
        
        Args:
            days_to_keep: Number of days to keep models
            
        Returns:
            Number of models deleted
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            deleted_count = 0
            
            for model_name in list(self.metadata.keys()):
                for version in list(self.metadata[model_name].keys()):
                    created_at = datetime.fromisoformat(self.metadata[model_name][version]['created_at'])
                    
                    if created_at < cutoff_date:
                        try:
                            self.delete_model(model_name, version)
                            deleted_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to delete old model {model_name} version {version}: {e}")
            
            logger.info(f"Cleaned up {deleted_count} old model versions")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old models: {e}")
            raise ModelError(f"Model cleanup failed: {e}")
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        try:
            total_models = 0
            total_size_mb = 0
            model_types = {}
            
            for model_name, versions in self.metadata.items():
                for version, metadata in versions.items():
                    total_models += 1
                    total_size_mb += metadata.get('file_size_mb', 0)
                    
                    model_type = metadata['model_type']
                    model_types[model_type] = model_types.get(model_type, 0) + 1
            
            return {
                'total_models': total_models,
                'total_size_mb': total_size_mb,
                'model_types': model_types,
                'unique_model_names': len(self.metadata),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get registry stats: {e}")
            raise ModelError(f"Registry stats failed: {e}")


# Singleton instance
_registry_instance = None

def get_model_registry() -> ModelRegistry:
    """Get model registry singleton instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance
