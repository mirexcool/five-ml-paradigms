SHELL := /bin/bash
.DEFAULT_GOAL := venv
VENV ?= venv
PYTHON ?= python3.12
.PHONY: venv install verify clean

venv:
	@test -x "$(VENV)/bin/python" || $(PYTHON) -m venv "$(VENV)"
	@$(VENV)/bin/python -m pip install --upgrade pip
	@$(MAKE) install

install:
	@$(VENV)/bin/python -m pip install -r requirements.txt
	@$(VENV)/bin/python -m pip check

verify:
	@$(VENV)/bin/python -c "import numpy,pandas,matplotlib,sklearn,jax,keras,pyBlindOpt,ipykernel,zstandard; print('Core imports OK')"
	@$(VENV)/bin/python -c "import sys; print(sys.executable,sys.version)"

clean:
	rm -rf "$(VENV)"
