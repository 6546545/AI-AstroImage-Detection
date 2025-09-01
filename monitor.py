#!/usr/bin/env python3
"""
Production Monitoring Script
Monitors system health, performance, and resource usage.
"""

import os
import sys
import time
import json
import psutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
import structlog

console = Console()


class SystemMonitor:
    """Production system monitor."""
    
    def __init__(self):
        self.logger = structlog.get_logger()
        self.metrics_file = Path("logs/metrics.json")
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self.start_time = datetime.now()
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # Network
            network = psutil.net_io_counters()
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'system': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_percent': disk.percent,
                    'disk_free_gb': disk.free / (1024**3)
                },
                'process': {
                    'memory_mb': process_memory.rss / (1024**2),
                    'cpu_percent': process.cpu_percent(),
                    'num_threads': process.num_threads()
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv
                },
                'uptime': (datetime.now() - self.start_time).total_seconds()
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error("Failed to get system metrics", error=str(e))
            return {}
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check system health status."""
        try:
            health_checks = {
                'models_exist': self._check_models(),
                'logs_writable': self._check_logs(),
                'disk_space': self._check_disk_space(),
                'memory_available': self._check_memory(),
                'system_test': self._run_system_test()
            }
            
            overall_health = all(health_checks.values())
            
            return {
                'healthy': overall_health,
                'checks': health_checks,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return {'healthy': False, 'error': str(e)}
    
    def _check_models(self) -> bool:
        """Check if model files exist."""
        model_files = [
            "models/astronomical_classifier.pth",
            "models/anomaly_detector.pth"
        ]
        return all(Path(f).exists() for f in model_files)
    
    def _check_logs(self) -> bool:
        """Check if logs directory is writable."""
        try:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            test_file = log_dir / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
            return True
        except Exception:
            return False
    
    def _check_disk_space(self) -> bool:
        """Check available disk space."""
        try:
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024**3)
            return free_gb > 1.0  # At least 1GB free
        except Exception:
            return False
    
    def _check_memory(self) -> bool:
        """Check available memory."""
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            return available_gb > 0.5  # At least 500MB free
        except Exception:
            return False
    
    def _run_system_test(self) -> bool:
        """Run quick system test."""
        try:
            result = subprocess.run([
                sys.executable, "space_analyzer.py", "test"
            ], capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        except Exception:
            return False
    
    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save metrics to file."""
        try:
            # Load existing metrics
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    existing_metrics = json.load(f)
            else:
                existing_metrics = {'history': []}
            
            # Add new metrics
            existing_metrics['history'].append(metrics)
            
            # Keep only last 1000 entries
            if len(existing_metrics['history']) > 1000:
                existing_metrics['history'] = existing_metrics['history'][-1000:]
            
            # Save updated metrics
            with open(self.metrics_file, 'w') as f:
                json.dump(existing_metrics, f, indent=2)
                
        except Exception as e:
            self.logger.error("Failed to save metrics", error=str(e))
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        try:
            if not self.metrics_file.exists():
                return {}
            
            with open(self.metrics_file, 'r') as f:
                metrics_data = json.load(f)
            
            history = metrics_data.get('history', [])
            if not history:
                return {}
            
            # Calculate statistics
            cpu_values = [m.get('system', {}).get('cpu_percent', 0) for m in history]
            memory_values = [m.get('system', {}).get('memory_percent', 0) for m in history]
            
            stats = {
                'cpu': {
                    'current': cpu_values[-1] if cpu_values else 0,
                    'average': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                    'max': max(cpu_values) if cpu_values else 0
                },
                'memory': {
                    'current': memory_values[-1] if memory_values else 0,
                    'average': sum(memory_values) / len(memory_values) if memory_values else 0,
                    'max': max(memory_values) if memory_values else 0
                },
                'samples': len(history)
            }
            
            return stats
            
        except Exception as e:
            self.logger.error("Failed to get performance stats", error=str(e))
            return {}
    
    def create_dashboard(self) -> Layout:
        """Create monitoring dashboard."""
        layout = Layout()
        
        # Get current data
        metrics = self.get_system_metrics()
        health = self.check_system_health()
        stats = self.get_performance_stats()
        
        # Create header
        header = Panel(
            f"[bold blue]Space Anomaly Detection System[/bold blue]\n"
            f"Monitoring Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            style="blue"
        )
        
        # Create metrics table
        metrics_table = Table(title="System Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")
        metrics_table.add_column("Status", style="yellow")
        
        if metrics:
            system = metrics.get('system', {})
            process = metrics.get('process', {})
            
            metrics_table.add_row("CPU Usage", f"{system.get('cpu_percent', 0):.1f}%", 
                                "🟢" if system.get('cpu_percent', 0) < 80 else "🔴")
            metrics_table.add_row("Memory Usage", f"{system.get('memory_percent', 0):.1f}%",
                                "🟢" if system.get('memory_percent', 0) < 80 else "🔴")
            metrics_table.add_row("Disk Usage", f"{system.get('disk_percent', 0):.1f}%",
                                "🟢" if system.get('disk_percent', 0) < 90 else "🔴")
            metrics_table.add_row("Process Memory", f"{process.get('memory_mb', 0):.1f} MB", "🟢")
            metrics_table.add_row("Uptime", f"{metrics.get('uptime', 0):.0f}s", "🟢")
        
        # Create health status
        health_status = "🟢 Healthy" if health.get('healthy', False) else "🔴 Unhealthy"
        health_panel = Panel(
            f"System Health: {health_status}\n"
            f"Models: {'✅' if health.get('checks', {}).get('models_exist', False) else '❌'}\n"
            f"Logs: {'✅' if health.get('checks', {}).get('logs_writable', False) else '❌'}\n"
            f"Disk: {'✅' if health.get('checks', {}).get('disk_space', False) else '❌'}\n"
            f"Memory: {'✅' if health.get('checks', {}).get('memory_available', False) else '❌'}\n"
            f"System Test: {'✅' if health.get('checks', {}).get('system_test', False) else '❌'}",
            title="Health Status"
        )
        
        # Create performance stats
        perf_panel = Panel(
            f"CPU Avg: {stats.get('cpu', {}).get('average', 0):.1f}%\n"
            f"CPU Max: {stats.get('cpu', {}).get('max', 0):.1f}%\n"
            f"Memory Avg: {stats.get('memory', {}).get('average', 0):.1f}%\n"
            f"Memory Max: {stats.get('memory', {}).get('max', 0):.1f}%\n"
            f"Samples: {stats.get('samples', 0)}",
            title="Performance Statistics"
        )
        
        # Layout
        layout.split_column(
            Layout(header, size=3),
            Layout(name="main"),
        )
        
        layout["main"].split_row(
            Layout(metrics_table),
            Layout(name="right")
        )
        
        layout["main"]["right"].split_column(
            Layout(health_panel),
            Layout(perf_panel)
        )
        
        return layout
    
    def run_monitoring(self, interval: int = 30) -> None:
        """Run continuous monitoring."""
        console.print("🚀 Starting production monitoring...", style="green")
        
        with Live(self.create_dashboard(), refresh_per_second=1) as live:
            while True:
                try:
                    # Get and save metrics
                    metrics = self.get_system_metrics()
                    self.save_metrics(metrics)
                    
                    # Update dashboard
                    live.update(self.create_dashboard())
                    
                    # Log significant events
                    if metrics:
                        system = metrics.get('system', {})
                        if system.get('cpu_percent', 0) > 80:
                            self.logger.warning("High CPU usage detected", cpu_percent=system['cpu_percent'])
                        if system.get('memory_percent', 0) > 80:
                            self.logger.warning("High memory usage detected", memory_percent=system['memory_percent'])
                    
                    time.sleep(interval)
                    
                except KeyboardInterrupt:
                    console.print("\n👋 Monitoring stopped", style="yellow")
                    break
                except Exception as e:
                    self.logger.error("Monitoring error", error=str(e))
                    time.sleep(interval)


def main():
    """Main monitoring function."""
    parser = argparse.ArgumentParser(description="Production monitoring script")
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    
    args = parser.parse_args()
    
    monitor = SystemMonitor()
    
    if args.once:
        # Run once
        metrics = monitor.get_system_metrics()
        health = monitor.check_system_health()
        stats = monitor.get_performance_stats()
        
        console.print(monitor.create_dashboard())
        
        # Save metrics
        monitor.save_metrics(metrics)
        
    else:
        # Run continuous monitoring
        monitor.run_monitoring(args.interval)


if __name__ == "__main__":
    main()
