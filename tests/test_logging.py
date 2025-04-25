import logging
from src.logging import configure_logging

def test_file_handler_attached(tmp_path):
    log_file = tmp_path / "app.log"
    configure_logging(str(log_file))

    file_handlers = [
        h for h in logging.root.handlers
        if isinstance(h, logging.FileHandler)
    ]
    assert file_handlers, "Expected a FileHandler"
    # verify it’s pointing to the right file
    assert file_handlers[0].baseFilename == str(log_file)

def test_stream_handler_attached(tmp_path):
    # file path is irrelevant here
    configure_logging(str(tmp_path / "unused.log"))

    stream_handlers = [
        h for h in logging.root.handlers
        if isinstance(h, logging.StreamHandler)
    ]
    assert stream_handlers, "Expected a StreamHandler"