import logging
import sys
import pytest
from src.logging import configure_logging

def teardown_function(function):
    logging.root.handlers.clear()

def test_file_and_stream_handlers_attached(tmp_path):
    log_file = tmp_path / "app.log"
    configure_logging(str(log_file))
    handlers = logging.root.handlers
    assert any(isinstance(h, logging.FileHandler) for h in handlers)
    assert any(isinstance(h, logging.StreamHandler) for h in handlers)
    fh = next(h for h in handlers if isinstance(h, logging.FileHandler))
    assert fh.baseFilename == str(log_file)

def test_clears_existing_handlers(tmp_path):
    dummy = logging.NullHandler()
    logging.root.addHandler(dummy)
    assert dummy in logging.root.handlers
    configure_logging(str(tmp_path / "a.log"))
    assert not any(isinstance(h, logging.NullHandler) for h in logging.root.handlers)
    assert len(logging.root.handlers) == 2

def test_handles_stdout_reconfigure_failure(monkeypatch, tmp_path):
    class BadStdout:
        def reconfigure(self, **kwargs):
            raise RuntimeError("oops")

    monkeypatch.setattr(sys, "stdout", BadStdout())
    configure_logging(str(tmp_path / "out.log"))
    handlers = logging.root.handlers
    assert any(isinstance(h, logging.FileHandler) for h in handlers)
    assert any(isinstance(h, logging.StreamHandler) for h in handlers)