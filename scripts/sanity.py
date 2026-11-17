"""Quick end-to-end check: a short Yearly run must clearly beat Naive2."""
import tempfile
from pathlib import Path

from deepbasis import evaluate
from deepbasis.train import RunConfig, train

root = Path(tempfile.mkdtemp())
fc = train(RunConfig("Yearly", "generic", "smape", 3, iterations=300), root=root, log_every=100)
sm, ma = evaluate.per_series_scores("Yearly", fc)
nsm, nma = evaluate.naive2_scores("Yearly")
print(f"300-iteration N-BEATS-G Yearly: sMAPE {sm.mean():.3f}  MASE {ma.mean():.3f}  (Naive2 {nsm.mean():.3f} / {nma.mean():.3f})")
