"""Continue the frozen follow-up plan after experiment 06A has completed."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments._runner import run_module

if __name__=='__main__': run_module('train_or_update.run_followup_suite',[])
