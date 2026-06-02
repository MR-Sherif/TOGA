from __future__ import annotations

import sys

import torch

from main import main


def _ensure_imagenet_config(argv):
    if "--config" in argv:
        return argv
    return [argv[0], "--config", "./configs/imagenet.yaml", *argv[1:]]


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    sys.argv = _ensure_imagenet_config(sys.argv)
    main()
