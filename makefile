setup: venv
	. .venv/bin/activate && python -m pip install --upgrade pip
	. .venv/bin/activate && pip install -r requirements_dev.txt

setup-win: venv
	. .venv/Scripts/activate && python -m pip install --upgrade pip
	. .venv/Scripts/activate && pip install -r requirements_dev.txt

venv:
	test -d .venv || python -m venv .venv

clean: clean-pyc
	rm -rf .venv

clean-pyc:
	find . -name "*.pyc" -exec rm -f {} + 
	find . -name "*.pyo" -exec rm -f {} +
	find . -name "*~" -exec rm -f {} +
	find . -name "__pycache__" -exec rm -fr {} +

# Run pipeline
pipeline:
	. .venv/bin/activate && python ./src/prep/preprocess.py


pipeline-win:
	. .venv/Scripts/activate && python ./src/prep/preprocess.py