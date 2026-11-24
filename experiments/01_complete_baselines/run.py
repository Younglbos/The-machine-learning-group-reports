"""Complete-data concatenation and individual-view K-means on four datasets."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments._runner import baseline

if __name__=='__main__': baseline(Path(__file__).resolve().parent,'complete')
