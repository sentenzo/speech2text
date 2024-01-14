PROJECT_DIR = speech2text

CODE = ${PROJECT_DIR} tests

run:
	poetry run python -m ${PROJECT_DIR}

init:
	poetry install

lint:
	poetry run isort ${CODE}
	poetry run black ${CODE}
	poetry run flake8 ${CODE} --count --select=E9,F63,F7,F82 --show-source --statistics

test:
	poetry run pytest -vsx -m "not slow"

test-all:
	poetry run pytest -vsx