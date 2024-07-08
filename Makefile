SHELL := /bin/bash

venv-llm:
	python3 -m venv venv-llm && \
		source ./venv-llm/bin/activate && \
		pip install --upgrade pip && \
		pip install poetry

install-llm: venv-llm
	source venv-llm/bin/activate && \
		poetry install
