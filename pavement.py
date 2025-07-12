#!/usr/bin/env python3

from paver.easy import task, needs, sh, cmdopts
import os
import shutil
import glob
import sys


@task
def setup():
    sh('pip install -r requirements.txt')
    sh('pip install -r dev-requirements.txt')

@task
def clean():
    for pattern in ['**/*.py[co]', '.coverage', '.coverage.*','coverage.xml']:
        for path in glob.glob(pattern, recursive=True):
            os.remove(path)

    targets = glob.glob('**/__pycache__', recursive=True) + ['htmlcov', '.pytest_cache']
    for d in targets:
        shutil.rmtree(d, ignore_errors=True)
@task
def test():
    sh(f'"{sys.executable}" -m pytest --cov=src --cov-report=term --cov-report=html --disable-warnings')


cmdopts(
    [
        ("vectorizer=", "v", "Text vectorizer to use: count or tfidf"),
        ("test-size=", "t", "Test set fraction (float, e.g. 0.2)"),
    ]
)
@task
def run(options):
    vec = options.get("vectorizer") or "tfidf"
    ts  = options.get("test_size") or 0.2
    sh(f"python -m main --vectorizer {vec} --test-size {ts}")

@task
@needs('setup', 'clean', 'test', 'run')
def default():
    pass
