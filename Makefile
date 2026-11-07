PYTHON ?= python3
.PHONY: install test cpp fixture validate smoke serve docker-build
install:
	$(PYTHON) -m pip install -e '.[dev]'
test:
	$(PYTHON) -m pytest -q
cpp:
	mkdir -p build
	g++ -std=c++17 -O2 -Wall -Wextra -Werror cpp/csi_features.cpp -o build/csi_features
fixture:
	crossguard generate-fixture --output datasets/synthetic --subjects 4
validate:
	crossguard validate --manifest datasets/synthetic/manifest.jsonl --data-root datasets/synthetic
smoke:
	$(PYTHON) scripts/smoke.py
serve:
	uvicorn crossguard_fi.serving.api:app --host 127.0.0.1 --port 8000
docker-build:
	docker build -t crossguard-fi:local .
