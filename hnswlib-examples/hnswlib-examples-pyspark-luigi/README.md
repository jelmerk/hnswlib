hnswlib-examples-pyspark-luigi
==============================

Create a new virtual environment luigi-venv:

    python3 -m venv luigi-venv

And activate the newly created virtual environment:

    . luigi-venv/bin/activate

Install dependencies:

    pip install wheel luigi requests

To execute the task you created, run the following command:

    python -m luigi --module flow Query --local-scheduler

