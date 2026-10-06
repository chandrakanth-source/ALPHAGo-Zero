"""
AlphaGo Zero - Master Curriculum Runner
======================================
This master script allows running any stage (Stages 1 through 5) or running all stages
sequentially from a single script.

Usage:
------
Run specific stage:
    python train_master_curriculum.py --stage 2
    python train_master_curriculum.py --stage 3
    python train_master_curriculum.py --stage 4
    python train_master_curriculum.py --stage 5

Run all remaining stages (Stages 2 through 5) sequentially:
    python train_master_curriculum.py --all
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train_curriculum import run_easy_level_training
from train_stage2_intermediate import run_intermediate_level_training
from train_stage3_difficult import run_difficult_level_training
from train_stage4_expert import run_expert_level_training
from train_stage5_super_expert import run_super_expert_level_training

STAGE_MAP = {
    1: ("Stage 1: Easy Level", run_easy_level_training),
    2: ("Stage 2: Intermediate Level", run_intermediate_level_training),
    3: ("Stage 3: Difficult Level", run_difficult_level_training),
    4: ("Stage 4: Expert Level", run_expert_level_training),
    5: ("Stage 5: Super Expert Level", run_super_expert_level_training),
}

def main():
    parser = argparse.ArgumentParser(description="AlphaGo Zero Master Curriculum Runner")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5], help="Run specific stage (1 to 5)")
    parser.add_argument("--all", action="store_true", help="Run all stages (2 to 5) sequentially")
    args = parser.parse_args()

    if args.all:
        print("\n=== RUNNING ALL REMAINING CURRICULUM STAGES (STAGES 2 - 5) ===")
        for s in [2, 3, 4, 5]:
            name, fn = STAGE_MAP[s]
            print(f"\n>>> Starting {name}...")
            fn()
    elif args.stage:
        name, fn = STAGE_MAP[args.stage]
        print(f"\n>>> Executing {name}...")
        fn()
    else:
        print("Please specify a stage to run. Example:")
        print("  python train_master_curriculum.py --stage 2")
        print("  python train_master_curriculum.py --all")

if __name__ == "__main__":
    main()
