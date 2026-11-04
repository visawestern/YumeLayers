#!/usr/bin/env python3
"""YumeLayers — точка входа. Запуск: python run.py [--port 7860] [--config config.json]"""
import sys, argparse
from src.server import main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    main(args.port, args.config)
