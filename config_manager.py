#!/usr/bin/env python3
"""
Production Configuration Manager
Handles loading and validating production configuration settings.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import structlog
from dataclasses import dataclass
from rich.console import Console
from rich.panel import Panel

console = Console()


@dataclass
class ModelConfig:
    """Model configuration settings."""
    path: str
    input_size: list
    num_classes: Optional[int] = None
    confidence_threshold: float = 0.8
    threshold: float = 0.15
    device: str = "auto"


@dataclass
class ProcessingConfig:
    """Processing configuration settings."""
    batch_size: int = 8
    num_workers: int = 4
    max_memory_usage: float = 0.8
    image_formats: list = None
    supported_classes: list = None


@dataclass
class OutputConfig:
    """Output configuration settings."""
    results_dir: str = "results"
    export_formats: list = None
    save_images: bool = True
    create_visualizations: bool = True
    compression_level: int = 6


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    file: str = "logs/space_analyzer.log"
    max_file_size: str = "100MB"
    backup_count: int = 5
    format: str = "json"
    include_timestamp: bool = True
    log_metrics: bool = True


class ProductionConfig:
    """Production configuration manager."""
    
    def __init__(self, config_path: str = "production_config.yaml"):
        self.config_path = Path(config_path)
        self.config_data = {}
        self.logger = None
        self._load_config()
        self._setup_logging()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config_data = yaml.safe_load(f)
                console.print(f"✅ Loaded configuration from {self.config_path}", style="green")
            else:
                console.print(f"⚠️  Configuration file {self.config_path} not found, using defaults", style="yellow")
                self._create_default_config()
        except Exception as e:
            console.print(f"❌ Error loading configuration: {e}", style="red")
            self._create_default_config()
    
    def _create_default_config(self) -> None:
        """Create default configuration."""
        self.config_data = {
            'system': {
                'name': 'Space Anomaly Detection & Classification System',
                'version': '1.0.0',
                'environment': 'production',
                'debug': False,
                'log_level': 'INFO'
            },
            'models': {
                'classifier': {
                    'path': 'models/astronomical_classifier.pth',
                    'input_size': [512, 512],
                    'num_classes': 10,
                    'confidence_threshold': 0.8,
                    'device': 'auto'
                },
                'anomaly_detector': {
                    'path': 'models/anomaly_detector.pth',
                    'input_size': [512, 512],
                    'threshold': 0.15,
                    'device': 'auto'
                }
            },
            'processing': {
                'batch_size': 8,
                'num_workers': 4,
                'max_memory_usage': 0.8,
                'image_formats': ['jpg', 'jpeg', 'png', 'tiff', 'fits'],
                'supported_classes': ['star', 'galaxy', 'nebula', 'planet', 'asteroid', 
                                    'comet', 'quasar', 'pulsar', 'black_hole', 'unknown']
            },
            'output': {
                'results_dir': 'results',
                'export_formats': ['json', 'csv', 'html'],
                'save_images': True,
                'create_visualizations': True,
                'compression_level': 6
            },
            'logging': {
                'file': 'logs/space_analyzer.log',
                'max_file_size': '100MB',
                'backup_count': 5,
                'format': 'json',
                'include_timestamp': True,
                'log_metrics': True
            }
        }
    
    def _setup_logging(self) -> None:
        """Setup structured logging."""
        try:
            log_config = self.get('logging', {})
            log_file = log_config.get('file', 'logs/space_analyzer.log')
            
            # Create logs directory if it doesn't exist
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            
            # Setup structured logging
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
            
            self.logger = structlog.get_logger()
            self.logger.info("Production configuration loaded", 
                           config_file=str(self.config_path),
                           environment=self.get('system.environment', 'production'))
            
        except Exception as e:
            console.print(f"❌ Error setting up logging: {e}", style="red")
            # Fallback to basic logging
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_model_config(self, model_type: str) -> ModelConfig:
        """Get model configuration."""
        model_data = self.get(f'models.{model_type}', {})
        return ModelConfig(**model_data)
    
    def get_processing_config(self) -> ProcessingConfig:
        """Get processing configuration."""
        processing_data = self.get('processing', {})
        return ProcessingConfig(**processing_data)
    
    def get_output_config(self) -> OutputConfig:
        """Get output configuration."""
        output_data = self.get('output', {})
        return OutputConfig(**output_data)
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        logging_data = self.get('logging', {})
        return LoggingConfig(**logging_data)
    
    def validate(self) -> bool:
        """Validate configuration."""
        try:
            # Check required directories
            required_dirs = [
                self.get('models.classifier.path', '').split('/')[0],
                self.get('models.anomaly_detector.path', '').split('/')[0],
                self.get('output.results_dir', 'results'),
                Path(self.get('logging.file', 'logs/space_analyzer.log')).parent
            ]
            
            for dir_path in required_dirs:
                if dir_path:
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
            
            # Validate model files exist
            classifier_path = self.get('models.classifier.path')
            anomaly_path = self.get('models.anomaly_detector.path')
            
            if not Path(classifier_path).exists():
                console.print(f"⚠️  Classifier model not found: {classifier_path}", style="yellow")
            
            if not Path(anomaly_path).exists():
                console.print(f"⚠️  Anomaly detector model not found: {anomaly_path}", style="yellow")
            
            self.logger.info("Configuration validation completed")
            return True
            
        except Exception as e:
            console.print(f"❌ Configuration validation failed: {e}", style="red")
            self.logger.error("Configuration validation failed", error=str(e))
            return False
    
    def print_summary(self) -> None:
        """Print configuration summary."""
        console.print(Panel.fit(
            f"[bold blue]Production Configuration Summary[/bold blue]\n"
            f"System: {self.get('system.name')} v{self.get('system.version')}\n"
            f"Environment: {self.get('system.environment')}\n"
            f"Log Level: {self.get('system.log_level')}\n"
            f"Batch Size: {self.get('processing.batch_size')}\n"
            f"Supported Classes: {len(self.get('processing.supported_classes', []))}\n"
            f"Results Directory: {self.get('output.results_dir')}",
            title="Configuration Status"
        ))


# Global configuration instance
config = ProductionConfig()


if __name__ == "__main__":
    # Test configuration loading
    config.print_summary()
    config.validate()
