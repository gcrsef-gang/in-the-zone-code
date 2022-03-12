# in-the-zone-code

Code repository for In The Zone, a research project for GCRSEF 2022. 

## Installation

```bash
pip install -e .
```

## Quickstart

Generate data:

```bash
python3 -m itz -v parse itz-data/
```

Create a SEM model:

```bash
python3 -m itz -v fit LONG_TERM itz-model.pickle itz-data/itz-data.csv --cov_math_path itz-model-cov-mat.csv
```

For more details:

```bash
python3 -m itz -h
```