"""Automate Kaggle dataset creation, kernel execution and result tracking."""

import os
import json
import time
import logging
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import sys
from datetime import datetime
import traceback
from typing import Optional

class KaggleLogger:
    """Custom logger for Kaggle automation."""
    
    def __init__(self, log_dir: str = "kaggle_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"kaggle_automation_{self.timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Create status file for tracking progress
        self.status_file = self.log_dir / f"status_{self.timestamp}.json"
        self.status = {
            "start_time": self.timestamp,
            "steps": {},
            "current_step": None,
            "errors": []
        }
        self._save_status()
    
    def log_step(self, step: str, status: str, details: Optional[dict] = None):
        """Log a step in the automation process."""
        self.status["current_step"] = step
        self.status["steps"][step] = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self._save_status()
        
        if status == "started":
            self.logger.info(f"Starting step: {step}")
        elif status == "completed":
            self.logger.info(f"Completed step: {step}")
        elif status == "failed":
            self.logger.error(f"Failed step: {step}")
            if details and "error" in details:
                self.logger.error(f"Error details: {details['error']}")
    
    def log_error(self, error: Exception, step: str):
        """Log an error with full traceback."""
        error_details = {
            "step": step,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }
        self.status["errors"].append(error_details)
        self._save_status()
        
        self.logger.error(f"Error in {step}: {str(error)}")
        self.logger.error(f"Traceback:\n{traceback.format_exc()}")
    
    def _save_status(self):
        """Save current status to JSON file."""
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)

class KaggleAutomation:
    def __init__(self, project_name: str = "fractalmemorynetwork"):
        self.project_name = project_name
        self.api = KaggleApi()
        self.api.authenticate()
        self.username = self.api.get_config_value('username')
        
        # Initialize logger
        self.logger = KaggleLogger()
        
        # Create necessary directories
        self.dataset_path = Path("kaggle_package")
        self.dataset_path.mkdir(exist_ok=True)
    
    def package_code(self):
        """Package all necessary code files into the Kaggle dataset directory."""
        step = "package_code"
        self.logger.log_step(step, "started")
        
        try:
            files_to_copy = [
                "kaggle.py",
                "config.py",
                "create_manual_data.py",
                "requirements.txt",
                "model/fmn.py",
                "model/__init__.py",
                "utils/data.py",
                "utils/__init__.py",
                "kaggle-metadata.json"
            ]
            
            # Create necessary subdirectories
            (self.dataset_path / "model").mkdir(exist_ok=True)
            (self.dataset_path / "utils").mkdir(exist_ok=True)
            
            copied_files = []
            missing_files = []
            
            # Copy files
            for file_path in files_to_copy:
                src = Path(file_path)
                dst = self.dataset_path / file_path
                
                if src.exists():
                    dst.parent.mkdir(exist_ok=True)
                    with open(src, 'r') as f:
                        content = f.read()
                    with open(dst, 'w') as f:
                        f.write(content)
                    copied_files.append(str(file_path))
                else:
                    missing_files.append(str(file_path))
            
            self.logger.log_step(step, "completed", {
                "copied_files": copied_files,
                "missing_files": missing_files
            })
            
        except Exception as e:
            self.logger.log_error(e, step)
            raise
    
    def create_or_update_dataset(self):
        """Create or update the Kaggle dataset."""
        step = "create_or_update_dataset"
        self.logger.log_step(step, "started")
        
        try:
            dataset_metadata = {
                "title": f"{self.project_name}",
                "id": f"{self.username}/{self.project_name}",
                "licenses": [{"name": "MIT"}]
            }
            
            # Write metadata file
            with open(self.dataset_path / "dataset-metadata.json", 'w') as f:
                json.dump(dataset_metadata, f, indent=2)
            
            # Create dataset ZIP file
            zip_path = f"{self.project_name}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                for file_path in self.dataset_path.rglob('*'):
                    if file_path.is_file():
                        zip_ref.write(file_path, file_path.relative_to(self.dataset_path))
            
            try:
                # Try to create new dataset
                self.api.dataset_create_new(
                    folder_path=str(self.dataset_path),
                    convert_to_csv=False,
                    dir_mode='zip'
                )
                status = "created"
            except:
                # Update existing dataset
                self.api.dataset_create_version(
                    folder_path=str(self.dataset_path),
                    version_notes=f"Update {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    convert_to_csv=False,
                    dir_mode='zip'
                )
                status = "updated"
            
            self.logger.log_step(step, "completed", {
                "dataset_id": f"{self.username}/{self.project_name}",
                "status": status,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.log_error(e, step)
            raise
    
    def create_or_update_kernel(self):
        """Create or update the Kaggle kernel."""
        step = "create_or_update_kernel"
        self.logger.log_step(step, "started")
        
        try:
            kernel_metadata = {
                "id": f"{self.username}/{self.project_name}-training",
                "title": f"{self.project_name} Training",
                "code_file": "kaggle.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": False,
                "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": [f"{self.username}/{self.project_name}"],
                "competition_sources": [],
                "kernel_sources": []
            }
            
            # Create kernel metadata file
            kernel_path = Path("kernel-metadata.json")
            with open(kernel_path, 'w') as f:
                json.dump(kernel_metadata, f, indent=2)
            
            try:
                # Try to create new kernel
                self.api.kernel_push(kernel_path.parent)
                status = "created"
            except:
                # Update existing kernel
                self.api.kernel_push_update(kernel_path.parent)
                status = "updated"
            
            self.logger.log_step(step, "completed", {
                "kernel_id": f"{self.username}/{self.project_name}-training",
                "status": status,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.log_error(e, step)
            raise
    
    def run_kernel_and_monitor(self):
        """Run the kernel and monitor its progress."""
        step = "run_kernel"
        self.logger.log_step(step, "started")
        
        try:
            self.api.kernel_pull_run()
            kernel_id = f"{self.username}/{self.project_name}-training"
            
            start_time = datetime.now()
            status_history = []
            
            while True:
                kernel_status = self.api.kernel_status(kernel_id)
                status = kernel_status['status']
                status_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "status": status
                })
                
                self.logger.log_step(step, "running", {
                    "current_status": status,
                    "runtime": str(datetime.now() - start_time),
                    "status_history": status_history
                })
                
                if status == 'complete':
                    self.logger.log_step(step, "completed", {
                        "final_status": "success",
                        "total_runtime": str(datetime.now() - start_time),
                        "status_history": status_history
                    })
                    break
                elif status in ['error', 'failed']:
                    error_details = {
                        "final_status": "failed",
                        "total_runtime": str(datetime.now() - start_time),
                        "status_history": status_history
                    }
                    self.logger.log_step(step, "failed", error_details)
                    raise Exception(f"Kernel failed with status: {status}")
                
                time.sleep(60)
                
        except Exception as e:
            self.logger.log_error(e, step)
            raise
    
    def download_results(self):
        """Download the training results."""
        step = "download_results"
        self.logger.log_step(step, "started")
        
        try:
            output_dir = Path("kaggle_results")
            output_dir.mkdir(exist_ok=True)
            
            self.api.kernel_output(
                f"{self.username}/{self.project_name}-training",
                path=str(output_dir)
            )
            
            # Log downloaded files
            downloaded_files = list(output_dir.rglob('*'))
            self.logger.log_step(step, "completed", {
                "output_dir": str(output_dir),
                "downloaded_files": [str(f.relative_to(output_dir)) for f in downloaded_files],
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.log_error(e, step)
            raise
    
    def run_complete_workflow(self):
        """Run the complete workflow from packaging to downloading results."""
        workflow_start = datetime.now()
        self.logger.log_step("workflow", "started", {
            "project_name": self.project_name,
            "username": self.username,
            "start_time": workflow_start.isoformat()
        })
        
        try:
            self.package_code()
            self.create_or_update_dataset()
            self.create_or_update_kernel()
            self.run_kernel_and_monitor()
            self.download_results()
            
            workflow_end = datetime.now()
            self.logger.log_step("workflow", "completed", {
                "total_runtime": str(workflow_end - workflow_start),
                "end_time": workflow_end.isoformat()
            })
            
        except Exception as e:
            self.logger.log_error(e, "workflow")
            raise

if __name__ == "__main__":
    # First, make sure you have your Kaggle API credentials in ~/.kaggle/kaggle.json
    if not Path.home().joinpath('.kaggle/kaggle.json').exists():
        print("Please set up your Kaggle API credentials first!")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Create new API token")
        print("3. Place kaggle.json in ~/.kaggle/")
        sys.exit(1)
    
    try:
        automation = KaggleAutomation()
        automation.run_complete_workflow()
    except Exception as e:
        print(f"\nAutomation failed! Check logs for details.")
        print(f"Error: {str(e)}")
        sys.exit(1) 