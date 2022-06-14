# Initiate environment setup
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

# Stage preprocess
preprocess:
	. .venv/bin/activate && python ./src/prep/preprocess.py

preprocess-win:
	. .venv/Scripts/activate && python ./src/prep/preprocess.py

# Stage train
train:
	. .venv/bin/activate && python ./src/train/train.py

train-win:
	. .venv/Scripts/activate && python ./src/train/train.py

# Stage test
test:
	. .venv/bin/activate && python ./src/test/test_models.py

test-win:
	. .venv/Scripts/activate && python ./src/test/test_models.py


# Stage animate
animate:
	. .venv/bin/activate && python ./src/animate/animate.py

animate-win:
	. .venv/Scripts/activate && python ./src/animate/animate.py
