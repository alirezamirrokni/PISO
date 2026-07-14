PYTHON ?= python
CONFIG ?= configs/reference.yaml
OUT ?= outputs/reference

.PHONY: install test run quick clean

install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest

run:
	$(PYTHON) -m ddp_pricing.cli run --config $(CONFIG) --output $(OUT)

quick:
	$(PYTHON) -m ddp_pricing.cli run --config configs/quick.yaml --output outputs/quick

clean:
	rm -rf outputs .pytest_cache build dist *.egg-info
