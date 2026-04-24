"""
generate_dataset.py
===================
Standalone script to generate the full train/val/test dataset.
Run this FIRST before training.

Usage:
  python generate_dataset.py
  python generate_dataset.py --no-augment          # seed data only
  python generate_dataset.py --verify              # print stats & examples
"""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset_builder import DatasetBuilder
from loguru import logger

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="config/config.yaml")
    parser.add_argument("--no-augment",  action="store_true")
    parser.add_argument("--verify",      action="store_true")
    args = parser.parse_args()

    logger.info("=" * 55)
    logger.info("  Hinglish + Manglish Dataset Generator")
    logger.info("=" * 55)

    builder = DatasetBuilder(config_path=args.config)
    splits  = builder.build_and_save(augment=not args.no_augment)
    builder.print_stats(splits)

    if args.verify:
        import random
        train = splits["train"]
        logger.info("\n3 Random Instruction-Formatted Examples:")
        for s in random.sample(train, min(3, len(train))):
            print(f"\n  Input  : {s['instruction_input']}")
            print(f"  Target : {s['instruction_target']}")
            print(f"  Emotion: {s['emotion']} | Category: {s['category']}")

    logger.success(
        f"\nDataset ready: "
        f"train={len(splits['train'])}, "
        f"val={len(splits['val'])}, "
        f"test={len(splits['test'])}\n"
        "Next step: python src/models/train.py"
    )
