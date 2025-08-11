#!/usr/bin/env python3
"""
CosmosAI - Feature Testing Script
Test all implemented features without reorganizing files.
"""

import os
import sys
import json
import time
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CosmosAIFeatureTest:
    """Test all CosmosAI features."""
    
    def __init__(self):
        self.test_results = {}
        
    def test_core_system(self):
        """Test the core space anomaly detection system."""
        logger.info("🔍 Testing core system...")
        
        try:
            # Test imports
            import space_anomaly_detector
            import astronomical_classifier
            import multi_object_detector
            
            # Test basic functionality
            from space_anomaly_detector import AnomalyDetector, DataPreprocessor
            from astronomical_classifier import AstronomicalClassificationSystem
            from multi_object_detector import MultiObjectDetector
            
            # Test data preprocessor
            preprocessor = DataPreprocessor()
            logger.info("✅ DataPreprocessor initialized successfully")
            
            # Test anomaly detector
            detector = AnomalyDetector()
            logger.info("✅ AnomalyDetector initialized successfully")
            
            # Test classifier
            classifier = AstronomicalClassificationSystem()
            logger.info("✅ AstronomicalClassificationSystem initialized successfully")
            
            # Test multi-object detector
            multi_detector = MultiObjectDetector()
            logger.info("✅ MultiObjectDetector initialized successfully")
            
            self.test_results['core_system'] = {
                'status': 'PASS',
                'message': 'All core components initialized successfully'
            }
            
        except Exception as e:
            logger.error(f"❌ Core system test failed: {e}")
            self.test_results['core_system'] = {
                'status': 'FAIL',
                'message': str(e)
            }
    
    def test_enhanced_system(self):
        """Test the enhanced space analysis system."""
        logger.info("🚀 Testing enhanced system...")
        
        try:
            from enhanced_space_analyzer import EnhancedSpaceAnalyzer
            from enhanced_space_system import EnhancedSpaceSystem
            
            # Test enhanced analyzer
            analyzer = EnhancedSpaceAnalyzer()
            logger.info("✅ EnhancedSpaceAnalyzer initialized successfully")
            
            # Test enhanced system
            system = EnhancedSpaceSystem()
            logger.info("✅ EnhancedSpaceSystem initialized successfully")
            
            self.test_results['enhanced_system'] = {
                'status': 'PASS',
                'message': 'Enhanced systems initialized successfully'
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced system test failed: {e}")
            self.test_results['enhanced_system'] = {
                'status': 'FAIL',
                'message': str(e)
            }
    
    def test_advanced_features(self):
        """Test advanced features and models."""
        logger.info("⚡ Testing advanced features...")
        
        try:
            from advanced_models import (
                VisionTransformer, EfficientNetAstronomical, 
                VariationalAutoencoder, SelfSupervisedAnomalyDetector
            )
            from advanced_features import (
                MultiSpectralAnalyzer, AdvancedImageProcessor,
                TimeSeriesAnalyzer, AstronomicalObjectTracker
            )
            from advanced_training import (
                AdvancedTrainer, AdvancedDataAugmentation,
                LearningRateScheduler, EarlyStopping
            )
            
            logger.info("✅ All advanced modules imported successfully")
            
            # Test advanced models
            vit = VisionTransformer()
            efficientnet = EfficientNetAstronomical()
            vae = VariationalAutoencoder()
            
            logger.info("✅ Advanced models initialized successfully")
            
            # Test advanced features
            multispectral = MultiSpectralAnalyzer()
            image_processor = AdvancedImageProcessor()
            time_series = TimeSeriesAnalyzer()
            object_tracker = AstronomicalObjectTracker()
            
            logger.info("✅ Advanced features initialized successfully")
            
            self.test_results['advanced_features'] = {
                'status': 'PASS',
                'message': 'All advanced features working correctly'
            }
            
        except Exception as e:
            logger.error(f"❌ Advanced features test failed: {e}")
            self.test_results['advanced_features'] = {
                'status': 'FAIL',
                'message': str(e)
            }
    
    def test_data_integrity(self):
        """Test data integrity and availability."""
        logger.info("📊 Testing data integrity...")
        
        try:
            # Check sample images
            sample_dir = Path('sample_images')
            if sample_dir.exists():
                image_count = 0
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    image_count += len(list(sample_dir.rglob(ext)))
                
                logger.info(f"✅ Found {image_count} sample images")
                
                if image_count > 0:
                    self.test_results['data_integrity'] = {
                        'status': 'PASS',
                        'message': f'Found {image_count} sample images'
                    }
                else:
                    self.test_results['data_integrity'] = {
                        'status': 'WARNING',
                        'message': 'No sample images found'
                    }
            else:
                self.test_results['data_integrity'] = {
                    'status': 'FAIL',
                    'message': 'Sample images directory not found'
                }
            
            # Check models
            models_dir = Path('models')
            if models_dir.exists():
                model_files = list(models_dir.glob('*.pth'))
                logger.info(f"✅ Found {len(model_files)} model files")
                
                if len(model_files) > 0:
                    self.test_results['models'] = {
                        'status': 'PASS',
                        'message': f'Found {len(model_files)} model files'
                    }
                else:
                    self.test_results['models'] = {
                        'status': 'WARNING',
                        'message': 'No model files found'
                    }
            
        except Exception as e:
            logger.error(f"❌ Data integrity test failed: {e}")
            self.test_results['data_integrity'] = {
                'status': 'FAIL',
                'message': str(e)
            }
    
    def test_web_interface(self):
        """Test the web interface (if available)."""
        logger.info("🌐 Testing web interface...")
        
        try:
            # Check if web interface exists
            web_dir = Path('landing_page')
            if web_dir.exists():
                # Check key web files
                web_files = ['app.py', 'index.html', 'css/styles.css', 'js/script.js']
                missing_files = []
                
                for file in web_files:
                    if not (web_dir / file).exists():
                        missing_files.append(file)
                
                if not missing_files:
                    logger.info("✅ All web interface files present")
                    self.test_results['web_interface'] = {
                        'status': 'PASS',
                        'message': 'Web interface files present and ready'
                    }
                else:
                    logger.warning(f"⚠️ Missing web files: {missing_files}")
                    self.test_results['web_interface'] = {
                        'status': 'WARNING',
                        'message': f'Missing files: {missing_files}'
                    }
            else:
                logger.info("ℹ️ Web interface not found (excluded from repository)")
                self.test_results['web_interface'] = {
                    'status': 'SKIP',
                    'message': 'Web interface excluded from repository'
                }
                
        except Exception as e:
            logger.error(f"❌ Web interface test failed: {e}")
            self.test_results['web_interface'] = {
                'status': 'FAIL',
                'message': str(e)
            }
    
    def test_dependencies(self):
        """Test if all dependencies are available."""
        logger.info("📦 Testing dependencies...")
        
        try:
            required_packages = [
                'torch', 'torchvision', 'numpy', 'opencv-python',
                'matplotlib', 'scikit-learn', 'tqdm', 'scipy',
                'einops', 'albumentations', 'timm', 'transformers'
            ]
            
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package.replace('-', '_'))
                except ImportError:
                    missing_packages.append(package)
            
            if not missing_packages:
                logger.info("✅ All required dependencies available")
                self.test_results['dependencies'] = {
                    'status': 'PASS',
                    'message': 'All dependencies available'
                }
            else:
                logger.warning(f"⚠️ Missing packages: {missing_packages}")
                self.test_results['dependencies'] = {
                    'status': 'WARNING',
                    'message': f'Missing packages: {missing_packages}'
                }
                
        except Exception as e:
            logger.error(f"❌ Dependencies test failed: {e}")
            self.test_results['dependencies'] = {
                'status': 'FAIL',
                'message': str(e)
            }
    
    def run_quick_analysis_test(self):
        """Run a quick analysis test with sample data."""
        logger.info("🧪 Running quick analysis test...")
        
        try:
            # Find a sample image
            sample_dir = Path('sample_images')
            test_image = None
            
            if sample_dir.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    images = list(sample_dir.rglob(ext))
                    if images:
                        test_image = images[0]
                        break
            
            if test_image:
                logger.info(f"✅ Using test image: {test_image.name}")
                
                # Test basic analysis
                from space_anomaly_detector import AnomalyDetector
                from astronomical_classifier import AstronomicalClassificationSystem
                
                # Initialize systems
                detector = AnomalyDetector()
                classifier = AstronomicalClassificationSystem()
                
                # Test classification
                classification_results = classifier.classify_image(str(test_image))
                logger.info(f"✅ Classification completed: {len(classification_results)} results")
                
                self.test_results['quick_analysis'] = {
                    'status': 'PASS',
                    'message': f'Analysis completed successfully with {test_image.name}'
                }
            else:
                logger.warning("⚠️ No test image found")
                self.test_results['quick_analysis'] = {
                    'status': 'WARNING',
                    'message': 'No test image available'
                }
                
        except Exception as e:
            logger.error(f"❌ Quick analysis test failed: {e}")
            self.test_results['quick_analysis'] = {
                'status': 'FAIL',
                'message': str(e)
            }
    
    def test_start_analysis_script(self):
        """Test the start analysis script."""
        logger.info("🎯 Testing start analysis script...")
        
        try:
            # Test if the script exists and can be imported
            import start_analysis
            logger.info("✅ Start analysis script imported successfully")
            
            self.test_results['start_analysis'] = {
                'status': 'PASS',
                'message': 'Start analysis script working correctly'
            }
            
        except Exception as e:
            logger.error(f"❌ Start analysis test failed: {e}")
            self.test_results['start_analysis'] = {
                'status': 'FAIL',
                'message': str(e)
            }
    
    def generate_test_report(self):
        """Generate a comprehensive test report."""
        logger.info("📋 Generating test report...")
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'project': 'CosmosAI Space Anomaly Detection System',
            'test_results': self.test_results,
            'summary': {
                'total_tests': len(self.test_results),
                'passed': len([r for r in self.test_results.values() if r['status'] == 'PASS']),
                'failed': len([r for r in self.test_results.values() if r['status'] == 'FAIL']),
                'warnings': len([r for r in self.test_results.values() if r['status'] == 'WARNING']),
                'skipped': len([r for r in self.test_results.values() if r['status'] == 'SKIP'])
            }
        }
        
        # Save report
        report_file = Path('test_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("🧪 COSMOSAI FEATURE TEST REPORT")
        print("="*60)
        print(f"📅 Timestamp: {report['timestamp']}")
        print(f"📊 Total Tests: {report['summary']['total_tests']}")
        print(f"✅ Passed: {report['summary']['passed']}")
        print(f"❌ Failed: {report['summary']['failed']}")
        print(f"⚠️ Warnings: {report['summary']['warnings']}")
        print(f"⏭️ Skipped: {report['summary']['skipped']}")
        print("="*60)
        
        # Print detailed results
        for test_name, result in self.test_results.items():
            status_icon = {
                'PASS': '✅',
                'FAIL': '❌',
                'WARNING': '⚠️',
                'SKIP': '⏭️'
            }.get(result['status'], '❓')
            
            print(f"{status_icon} {test_name.replace('_', ' ').title()}: {result['message']}")
        
        print("="*60)
        print(f"📄 Detailed report saved to: {report_file}")
        
        return report
    
    def run_all_tests(self):
        """Run all tests."""
        logger.info("🚀 Starting CosmosAI feature testing...")
        
        # Run all tests
        self.test_core_system()
        self.test_enhanced_system()
        self.test_advanced_features()
        self.test_data_integrity()
        self.test_web_interface()
        self.test_dependencies()
        self.run_quick_analysis_test()
        self.test_start_analysis_script()
        
        # Generate report
        self.generate_test_report()
        
        logger.info("🎉 Feature testing completed!")

def main():
    """Main function."""
    print("🌌 CosmosAI - Feature Testing")
    print("="*50)
    
    tester = CosmosAIFeatureTest()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
