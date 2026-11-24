"""Three K-means stream baselines with and without drift."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments._runner import baseline

if __name__=='__main__': baseline(Path(__file__).resolve().parent,'b')
