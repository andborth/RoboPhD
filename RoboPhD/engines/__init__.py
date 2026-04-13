"""Engine implementations for optimize_anything().

Each engine module provides a run function that accepts an evaluator,
dataset, seed candidate, and engine-specific config, returning an
OptimizeResult.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def build_val_split(
    dataset: List[Dict],
    val_dataset: Optional[List[Dict]],
    val_size: int,
    seed: int,
) -> tuple[List[Dict], List[Dict]]:
    """Build train/val split for validate-then-select engines (GEPA, Autoresearch).

    When val_dataset is provided:
      - Sample val_size examples from it for validation
      - Add remaining val_dataset examples to the training pool
    When val_dataset is None:
      - Split dataset into train/val using val_size
    """
    rng = random.Random(seed)

    if val_dataset is not None:
        if len(val_dataset) <= val_size:
            valset = list(val_dataset)
            trainset = list(dataset)
        else:
            shuffled_val = list(val_dataset)
            rng.shuffle(shuffled_val)
            valset = shuffled_val[:val_size]
            remaining = shuffled_val[val_size:]
            trainset = list(dataset) + remaining
        logger.info(
            f"Val split: {len(valset)} val from val_dataset "
            f"({len(val_dataset)} provided), {len(trainset)} train"
        )
    else:
        shuffled = list(dataset)
        rng.shuffle(shuffled)
        valset = shuffled[:val_size]
        trainset = shuffled[val_size:]
        logger.info(
            f"Val split: {len(valset)} val, {len(trainset)} train "
            f"(split from {len(dataset)} dataset)"
        )

    return trainset, valset
