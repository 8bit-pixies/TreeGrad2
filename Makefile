test-python:
	poetry run python -m pytest tests

lint-python:
	# poetry run python -m mypy treegrad/ tests/
	poetry run python -m isort treegrad/ tests/
	poetry run python -m flake8 treegrad/ tests/
	poetry run python -m black --check treegrad tests

format-python:
	poetry run python -m isort treegrad/ tests/
	poetry run python -m black --target-version py37 treegrad tests

