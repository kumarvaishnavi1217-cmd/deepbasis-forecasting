PY ?= python

.PHONY: install data test lint naive2-check baselines train-laptop train-paper evaluate figures

install:            ## install package + dev tools
	$(PY) -m pip install -e ".[dev,baselines]"

data:               ## download M4 (~250 MB)
	$(PY) -m deepbasis.cli download

test:
	$(PY) -m pytest -q

lint:
	ruff check .

baselines:          ## Theta + SES with statsforecast (~10 min on a laptop)
	$(PY) -m deepbasis.baselines

train-laptop:       ## 6-model N-BEATS-G ensemble per frequency (~4 h on an M1)
	$(PY) -m deepbasis.cli grid --profile laptop --arch generic

train-paper:        ## full paper protocol: 180 models x 6 frequencies (needs a GPU cluster)
	$(PY) -m deepbasis.cli grid --profile paper --arch generic
	$(PY) -m deepbasis.cli grid --profile paper --arch interpretable

evaluate:           ## results/RESULTS.md + results/results.json + figures
	$(PY) -m deepbasis.cli evaluate

figures:
	$(PY) scripts/plot_decomposition.py --freq Monthly
