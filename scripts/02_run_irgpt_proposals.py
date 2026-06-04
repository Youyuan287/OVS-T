#!/usr/bin/env python3
"""Compatibility wrapper for the renamed prompt proposal builder."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).with_name("02_build_prompt_proposals.py")
    runpy.run_path(str(target), run_name="__main__")
