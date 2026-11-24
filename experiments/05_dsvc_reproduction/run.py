"""Pinned DSVC entry point; training verification status is in README.md."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments._runner import paper

if __name__=='__main__': paper(Path(__file__).resolve().parent,'train_or_update.run_dsvc')
