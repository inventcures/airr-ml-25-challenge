import sys
import subprocess
import logging
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("run_full_pipeline.log")
    ]
)
logger = logging.getLogger("Pipeline")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
STATE_FILE = PROJECT_ROOT / "pipeline_state.json"

class PipelineManager:
    def __init__(self):
        self.state = self.load_state()
        
    def load_state(self) -> Dict:
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load state file: {e}")
        return {"completed_steps": []}
        
    def save_state(self):
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state file: {e}")
            
    def mark_completed(self, step_name: str):
        if step_name not in self.state["completed_steps"]:
            self.state["completed_steps"].append(step_name)
            self.save_state()
            logger.info(f"💾 Checkpoint saved: '{step_name}' marked as complete.")
            
    def is_completed(self, step_name: str) -> bool:
        return step_name in self.state["completed_steps"]

    def run_step(self, step_name: str, command: List[str], description: str, always_run: bool = False):
        """
        Runs a step if it hasn't been completed yet.
        """
    def run_step(self, step_name: str, command: List[str], description: str, force_rerun: bool = False):
        """
        Runs a step.
        """
        # 1. Check if we should skip this step entirely? (Handled by caller or here?)
        # For simplicity, we'll let the caller handle 'only' logic or handle it here if we pass args.
        
        # 2. Check completion
        if self.is_completed(step_name) and not force_rerun:
            logger.info(f"⏭️  Skipping previously completed step: {description}")
            return

        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Starting Step: {description}")
        logger.info(f"   Command: {' '.join(command)}")
        logger.info(f"{'='*60}\n")
        
        try:
            # sys.executable ensures we use the same python env
            if command[0] == "python":
                command[0] = sys.executable
                
            # Run command and stream output
            result = subprocess.run(command, check=True, cwd=PROJECT_ROOT)
            
            # If successful, mark as complete
            self.mark_completed(step_name)
            logger.info(f"✅ Step Completed: {description}\n")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Step Failed: {description}")
            logger.error(f"   Exit Code: {e.returncode}")
            sys.exit(1)
        except KeyboardInterrupt:
            logger.warning("\n⚠️  Pipeline interrupted by user.")
            sys.exit(130)

    def clean_partial_outputs(self):
        """
        Moves partial files to trash.
        Only runs if not marked as completed.
        """
        step_name = "clean_partials"
        if self.is_completed(step_name):
            logger.info("⏭️  Skipping cleanup (already done).")
            return
            
        logger.info("🕵️  Scanning outputs for partial/debug files...")
        
        patterns = ["*_preds.csv", "*_oof.csv", "*_ranking.csv"]
        deleted_count = 0
        kept_count = 0
        
        trash_dir = OUTPUTS_DIR / "trash_partial"
        trash_dir.mkdir(exist_ok=True)
        
        if not OUTPUTS_DIR.exists():
            return
            
        for f in OUTPUTS_DIR.rglob("*"):
            if not f.is_file(): continue
            
            # CRITICAL: Do NOT touch .joblib or .pt model files
            if not any(f.match(p) for p in patterns):
                continue
                
            try:
                line_count = sum(1 for _ in open(f, 'rb'))
                
                # Threshold: 1000 lines (debug sets are ~400)
                if line_count < 1000:
                    logger.warning(f"  ⚠️ Moving partial file ({line_count} lines) to trash: {f.name}")
                    shutil.move(str(f), str(trash_dir / f.name))
                    deleted_count += 1
                else:
                    logger.info(f"  ✅ Keeping valid file ({line_count} lines): {f.relative_to(PROJECT_ROOT)}")
                    kept_count += 1
            except Exception as e:
                logger.error(f"  Error checking {f}: {e}")

        logger.info(f"Cleanup complete. Moved {deleted_count} files to trash.")
        self.mark_completed(step_name)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RunPod Full Pipeline Orchestrator")
    parser.add_argument("--rerun", nargs="+", help="List of steps to force rerun (e.g. build_datasets)")
    parser.add_argument("--only", nargs="+", help="Run ONLY these steps (skip others)")
    parser.add_argument("--list", action="store_true", help="List all available steps")
    args = parser.parse_args()

    manager = PipelineManager()
    
    # Define steps structure: (name, command, description)
    steps = [
        ("build_datasets", ["python", "data/load_all_datasets.py", "--force"], "Build Dataset Pickles"),
        ("train_stats", ["python", "malid/train_stats_all.py"], "Stats Model (Train & Preds)"),
        ("train_esm", ["python", "malid/train_esm_seq_all.py"], "ESM Sequence Model"),
        ("train_deeprc_cv", ["python", "deeprc/train_mil_cv.py"], "DeepRC Training (CV)"),
        ("infer_deeprc_cv", ["python", "deeprc/infer_mil_cv.py"], "DeepRC Inference"),
        ("cluster_lancedb", ["python", "malid/run_clustering_lancedb.py"], "Clustering (LanceDB)"),
        ("rank_task2", ["python", "scripts/rank_sequences_task2_all.py"], "Task 2 Sequence Ranking"),
        ("meta_ensemble", ["python", "malid/train_meta_and_predict.py"], "Meta-Ensemble Prediction"),
        ("build_submission", ["python", "scripts/build_submission.py"], "Build Final Submission"),
    ]
    
    if args.list:
        print("Available steps:")
        for name, _, desc in steps:
            print(f"  - {name}: {desc}")
        return

    logger.info("========================================")
    logger.info("   RunPod Full Pipeline Orchestrator    ")
    logger.info("========================================")
    logger.info(f"Current State: {len(manager.state['completed_steps'])} steps completed.")
    
    if args.rerun:
        logger.info(f"🔄 Forcing rerun for: {args.rerun}")
    if args.only:
        logger.info(f"🔒 limiting execution to: {args.only}")
    
    # 1. Cleanup (Always run unless filtered out by --only?)
    # Usually cleanup is good. Let's run it unless --only is specific.
    if not args.only or "cleanup" in args.only:
        manager.clean_partial_outputs()
    
    # 2. Pipeline Steps
    for name, cmd, desc in steps:
        # Check if we should skip
        if args.only and name not in args.only:
            continue
            
        # Check if we should force rerun
        force = False
        if args.rerun and name in args.rerun:
            force = True
            
        manager.run_step(name, cmd, desc, force_rerun=force)
    
    logger.info("\n🎉 Pipeline Execution Finished.")

if __name__ == "__main__":
    main()
