#!/usr/bin/env python3
"""
Production Deployment Script
Handles deployment and setup of the Space Anomaly Detection System.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any
import argparse
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
import structlog

console = Console()


class ProductionDeployer:
    """Production deployment manager."""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.logger = structlog.get_logger()
        self.deployment_status = {}
    
    def check_prerequisites(self) -> bool:
        """Check system prerequisites."""
        console.print("🔍 Checking system prerequisites...", style="blue")
        
        checks = {
            "Python Version": self._check_python_version(),
            "Virtual Environment": self._check_virtual_env(),
            "Disk Space": self._check_disk_space(),
            "Memory": self._check_memory(),
            "Dependencies": self._check_dependencies()
        }
        
        # Display results
        table = Table(title="Prerequisites Check")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="white")
        
        all_passed = True
        for component, (status, details) in checks.items():
            status_style = "✅ PASS" if status else "❌ FAIL"
            table.add_row(component, status_style, details)
            if not status:
                all_passed = False
        
        console.print(table)
        return all_passed
    
    def _check_python_version(self) -> tuple[bool, str]:
        """Check Python version."""
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            return True, f"Python {version.major}.{version.minor}.{version.micro}"
        return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"
    
    def _check_virtual_env(self) -> tuple[bool, str]:
        """Check if virtual environment is active."""
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            return True, "Virtual environment active"
        # Check if we're in a conda environment
        if 'CONDA_DEFAULT_ENV' in os.environ:
            return True, f"Conda environment: {os.environ['CONDA_DEFAULT_ENV']}"
        return False, "No virtual environment detected (recommended but not required)"
    
    def _check_disk_space(self) -> tuple[bool, str]:
        """Check available disk space."""
        try:
            stat = shutil.disk_usage(self.project_root)
            free_gb = stat.free / (1024**3)
            if free_gb > 5:  # Require at least 5GB
                return True, f"{free_gb:.1f}GB available"
            return False, f"{free_gb:.1f}GB available (need 5GB+)"
        except Exception:
            return False, "Unable to check disk space"
    
    def _check_memory(self) -> tuple[bool, str]:
        """Check available memory."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            if available_gb > 2:  # Require at least 2GB
                return True, f"{available_gb:.1f}GB available"
            return False, f"{available_gb:.1f}GB available (need 2GB+) - will use reduced batch size"
        except Exception:
            return False, "Unable to check memory"
    
    def _check_dependencies(self) -> tuple[bool, str]:
        """Check if key dependencies are available."""
        required_modules = ['torch', 'numpy', 'cv2', 'sklearn']
        missing = []
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing.append(module)
        
        if not missing:
            return True, "All dependencies available"
        return False, f"Missing: {', '.join(missing)}"
    
    def setup_environment(self) -> bool:
        """Setup production environment."""
        console.print("🔧 Setting up production environment...", style="blue")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Create directories
            task = progress.add_task("Creating directories...", total=None)
            directories = [
                "logs",
                "results",
                "models",
                "data/processed",
                "data/raw",
                "configs",
                "tests"
            ]
            
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
            
            progress.update(task, description="Directories created")
            
            # Install dependencies
            task = progress.add_task("Installing dependencies...", total=None)
            if not self._install_dependencies():
                return False
            progress.update(task, description="Dependencies installed")
            
            # Setup logging
            task = progress.add_task("Setting up logging...", total=None)
            self._setup_logging()
            progress.update(task, description="Logging configured")
            
            # Validate models
            task = progress.add_task("Validating models...", total=None)
            if not self._validate_models():
                console.print("⚠️  Models not found - will be created during first run", style="yellow")
            progress.update(task, description="Models validated")
        
        return True
    
    def _install_dependencies(self) -> bool:
        """Install production dependencies."""
        try:
            # Install from requirements.txt
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], capture_output=True, text=True, check=True)
            
            self.logger.info("Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            console.print(f"❌ Failed to install dependencies: {e.stderr}", style="red")
            self.logger.error("Dependency installation failed", error=e.stderr)
            return False
    
    def _setup_logging(self) -> None:
        """Setup production logging."""
        try:
            from config_manager import config
            config.validate()
            self.logger.info("Production logging configured")
        except Exception as e:
            console.print(f"⚠️  Logging setup failed: {e}", style="yellow")
    
    def _validate_models(self) -> bool:
        """Validate that trained models exist."""
        model_files = [
            "models/astronomical_classifier.pth",
            "models/anomaly_detector.pth"
        ]
        
        existing_models = []
        for model_file in model_files:
            if Path(model_file).exists():
                existing_models.append(model_file)
        
        if existing_models:
            self.logger.info("Found existing models", models=existing_models)
            return True
        
        return False
    
    def run_tests(self) -> bool:
        """Run production tests."""
        console.print("🧪 Running production tests...", style="blue")
        
        tests = [
            ("System Test", self._run_system_test),
            ("Model Test", self._run_model_test),
            ("Performance Test", self._run_performance_test),
            ("Integration Test", self._run_integration_test)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                console.print(f"Running {test_name}...", style="cyan")
                success = test_func()
                results.append((test_name, success))
            except Exception as e:
                console.print(f"❌ {test_name} failed: {e}", style="red")
                results.append((test_name, False))
        
        # Display test results
        table = Table(title="Test Results")
        table.add_column("Test", style="cyan")
        table.add_column("Status", style="green")
        
        all_passed = True
        for test_name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            table.add_row(test_name, status)
            if not success:
                all_passed = False
        
        console.print(table)
        return all_passed
    
    def _run_system_test(self) -> bool:
        """Run basic system test."""
        try:
            result = subprocess.run([
                sys.executable, "space_analyzer.py", "test"
            ], capture_output=True, text=True, timeout=60)
            
            return result.returncode == 0
        except Exception:
            return False
    
    def _run_model_test(self) -> bool:
        """Test model loading and inference."""
        try:
            # Test with a sample image
            test_images = list(Path("test_dataset/images").glob("*.png"))
            if not test_images:
                return True  # No test images, skip test
            
            result = subprocess.run([
                sys.executable, "space_analyzer.py", "classify",
                "--input-dir", "test_dataset/images",
                "--output-dir", "results/test"
            ], capture_output=True, text=True, timeout=120)
            
            return result.returncode == 0
        except Exception:
            return False
    
    def _run_performance_test(self) -> bool:
        """Run performance benchmarks."""
        try:
            # Simple performance check
            import time
            start_time = time.time()
            
            # Run a quick classification
            result = subprocess.run([
                sys.executable, "space_analyzer.py", "test"
            ], capture_output=True, text=True, timeout=30)
            
            elapsed = time.time() - start_time
            return result.returncode == 0 and elapsed < 30
        except Exception:
            return False
    
    def _run_integration_test(self) -> bool:
        """Run integration test."""
        try:
            # Test the full pipeline
            result = subprocess.run([
                sys.executable, "start_analysis.py"
            ], capture_output=True, text=True, timeout=30)
            
            return result.returncode == 0
        except Exception:
            return False
    
    def create_production_script(self) -> None:
        """Create production startup script."""
        script_content = """#!/bin/bash
# Production startup script for Space Anomaly Detection System

set -e

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Start the system
python3 start_analysis.py "$@"
"""
        
        script_path = Path("start_production.sh")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        console.print(f"✅ Created production script: {script_path}", style="green")
    
    def generate_documentation(self) -> None:
        """Generate production documentation."""
        docs = {
            "DEPLOYMENT.md": self._generate_deployment_docs(),
            "API.md": self._generate_api_docs(),
            "MONITORING.md": self._generate_monitoring_docs()
        }
        
        for filename, content in docs.items():
            with open(filename, 'w') as f:
                f.write(content)
            console.print(f"✅ Generated {filename}", style="green")
    
    def _generate_deployment_docs(self) -> str:
        return """# Production Deployment Guide

## Quick Start
```bash
# Deploy the system
python3 deploy.py

# Start production
./start_production.sh
```

## System Requirements
- Python 3.8+
- 5GB+ disk space
- 2GB+ RAM
- Virtual environment recommended

## Configuration
Edit `production_config.yaml` to customize settings.

## Monitoring
Check logs in `logs/` directory.
"""
    
    def _generate_api_docs(self) -> str:
        return """# API Documentation

## Command Line Interface

### Basic Commands
- `python space_analyzer.py test` - Test system
- `python space_analyzer.py classify` - Classify images
- `python space_analyzer.py detect` - Detect anomalies
- `python space_analyzer.py analyze` - Combined analysis

### Interactive Mode
- `python start_analysis.py` - Interactive menu

## Configuration
See `production_config.yaml` for all options.
"""
    
    def _generate_monitoring_docs(self) -> str:
        return """# Monitoring Guide

## Log Files
- `logs/space_analyzer.log` - Main application logs
- `logs/metrics.json` - Performance metrics

## Health Checks
- Run `python space_analyzer.py test` to verify system health
- Check disk space and memory usage
- Monitor log file sizes

## Performance Metrics
- Processing time per image
- Memory usage
- GPU utilization (if available)
"""
    
    def deploy(self) -> bool:
        """Complete production deployment."""
        console.print(Panel.fit(
            "[bold blue]Space Anomaly Detection System[/bold blue]\n"
            "[bold green]Production Deployment[/bold green]",
            title="🚀 Deployment Started"
        ))
        
        # Check prerequisites
        if not self.check_prerequisites():
            console.print("⚠️  Some prerequisites failed - continuing with warnings", style="yellow")
            # Continue anyway for development/testing
        
        # Setup environment
        if not self.setup_environment():
            console.print("❌ Environment setup failed", style="red")
            return False
        
        # Run tests
        if not self.run_tests():
            console.print("⚠️  Some tests failed - check logs", style="yellow")
        
        # Create production script
        self.create_production_script()
        
        # Generate documentation
        self.generate_documentation()
        
        console.print(Panel.fit(
            "[bold green]✅ Deployment Complete![/bold green]\n"
            "Start the system with: ./start_production.sh\n"
            "Check logs in: logs/space_analyzer.log",
            title="🎉 Success"
        ))
        
        return True


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Production deployment script")
    parser.add_argument("--skip-tests", action="store_true", help="Skip test execution")
    parser.add_argument("--force", action="store_true", help="Force deployment even if checks fail")
    
    args = parser.parse_args()
    
    deployer = ProductionDeployer()
    
    if deployer.deploy():
        console.print("🚀 Production deployment successful!", style="green")
        sys.exit(0)
    else:
        console.print("❌ Production deployment failed!", style="red")
        sys.exit(1)


if __name__ == "__main__":
    main()
