import logging
import os
import pytest
from unittest.mock import patch, Mock
from src.utils.logging_config import setup_logging, get_logger


class TestLoggingConfig:
    def test_setup_logging_defaults_level(self):
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        setup_logging()
        assert root_logger.level == logging.INFO

    def test_setup_logging_defaults_console_handlers(self):
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        setup_logging()
        console_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(console_handlers) >= 1

    def test_setup_logging_defaults_file_handlers(self):
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        setup_logging()
        file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 0

    def test_setup_logging_file_path_total_handlers(self, tmp_path):
        log_file = tmp_path / "test.log"
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        setup_logging(log_file_path=str(log_file))
        assert len(root_logger.handlers) >= 2

    def test_setup_logging_file_path_file_handlers(self, tmp_path):
        log_file = tmp_path / "test.log"
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        setup_logging(log_file_path=str(log_file))
        file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) >= 1

    def test_setup_logging_file_path_console_handlers(self, tmp_path):
        log_file = tmp_path / "test.log"
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        setup_logging(log_file_path=str(log_file))
        console_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(console_handlers) >= 1

    def test_setup_logging_file_path_exists(self, tmp_path):
        log_file = tmp_path / "test.log"
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        setup_logging(log_file_path=str(log_file))
        assert log_file.exists()

    def test_setup_logging_custom_level_level(self):
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        setup_logging(level='DEBUG')
        assert root_logger.level == logging.DEBUG

    def test_setup_logging_custom_level_handlers(self):
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        setup_logging(level='DEBUG')
        assert len(root_logger.handlers) >= 1

    def test_setup_logging_creates_directory_parent(self, tmp_path):
        log_file = tmp_path / "nonexistent" / "subdir" / "log.txt"
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        setup_logging(log_file_path=str(log_file))
        assert log_file.parent.exists()

    def test_setup_logging_creates_directory_file(self, tmp_path):
        log_file = tmp_path / "nonexistent" / "subdir" / "log.txt"
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        setup_logging(log_file_path=str(log_file))
        assert log_file.exists()

    def test_setup_logging_clears_handlers_present(self):
        root_logger = logging.getLogger()
        dummy_handler = logging.NullHandler()
        root_logger.addHandler(dummy_handler)
        assert dummy_handler in root_logger.handlers

    def test_setup_logging_clears_handlers_removed(self):
        root_logger = logging.getLogger()
        dummy_handler = logging.NullHandler()
        root_logger.addHandler(dummy_handler)
        setup_logging()
        assert dummy_handler not in root_logger.handlers

    def test_setup_logging_clears_handlers_new(self):
        root_logger = logging.getLogger()
        dummy_handler = logging.NullHandler()
        root_logger.addHandler(dummy_handler)
        setup_logging()
        assert len(root_logger.handlers) >= 1

    def test_setup_logging_formatter_not_none(self):
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        setup_logging()
        for handler in root_logger.handlers:
            assert handler.formatter is not None

    def test_setup_logging_formatter_asctime(self):
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        setup_logging()
        for handler in root_logger.handlers:
            format_string = handler.formatter._fmt
            assert '%(asctime)s' in format_string

    def test_setup_logging_formatter_levelname(self):
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        setup_logging()
        for handler in root_logger.handlers:
            format_string = handler.formatter._fmt
            assert '%(levelname)s' in format_string

    def test_setup_logging_formatter_message(self):
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        setup_logging()
        for handler in root_logger.handlers:
            format_string = handler.formatter._fmt
            assert '%(message)s' in format_string

    def test_setup_logging_file_handler_encoding_handlers(self, tmp_path):
        log_file = tmp_path / "test.log"
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        setup_logging(log_file_path=str(log_file))
        file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) >= 1

    def test_setup_logging_file_handler_encoding_content(self, tmp_path):
        log_file = tmp_path / "test.log"
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        setup_logging(log_file_path=str(log_file))
        file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
        file_handler = file_handlers[0]
        test_message = "Test message with unicode: café"
        file_handler.emit(logging.LogRecord(name="test", level=logging.INFO, pathname="", lineno=0, msg=test_message, args=(), exc_info=None))
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Test message with unicode: café" in content

    def test_setup_logging_multiple_calls_handler_count(self, tmp_path):
        log_file1 = tmp_path / "test1.log"
        log_file2 = tmp_path / "test2.log"
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        setup_logging(log_file_path=str(log_file1))
        initial_handlers = len(root_logger.handlers)
        setup_logging(log_file_path=str(log_file2))
        assert len(root_logger.handlers) == initial_handlers

    def test_setup_logging_multiple_calls_file_handlers(self, tmp_path):
        log_file1 = tmp_path / "test1.log"
        log_file2 = tmp_path / "test2.log"
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        setup_logging(log_file_path=str(log_file1))
        setup_logging(log_file_path=str(log_file2))
        file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) >= 1

    def test_setup_logging_multiple_calls_filename(self, tmp_path):
        log_file1 = tmp_path / "test1.log"
        log_file2 = tmp_path / "test2.log"
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        setup_logging(log_file_path=str(log_file1))
        setup_logging(log_file_path=str(log_file2))
        file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
        assert log_file2.name in file_handlers[0].baseFilename

    def test_get_logger_type(self):
        logger = get_logger("test_logger")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_name(self):
        logger = get_logger("test_logger")
        assert logger.name == "test_logger"

    def test_get_logger_level(self):
        logger = get_logger("test_logger")
        root_logger = logging.getLogger()
        assert logger.level in [root_logger.level, logging.NOTSET]

    def test_get_logger_multiple_calls_same(self):
        logger1 = get_logger("test_logger")
        logger2 = get_logger("test_logger")
        assert logger1 is logger2

    def test_get_logger_different_names_not_same(self):
        logger1 = get_logger("logger1")
        logger2 = get_logger("logger2")
        assert logger1 is not logger2

    def test_get_logger_different_names_name1(self):
        logger1 = get_logger("logger1")
        logger2 = get_logger("logger2")
        assert logger1.name == "logger1"

    def test_get_logger_different_names_name2(self):
        logger1 = get_logger("logger1")
        logger2 = get_logger("logger2")
        assert logger2.name == "logger2"

    @patch('src.utils.logging_config.config')
    def test_setup_logging_config_defaults_level(self, mock_config):
        mock_config.LOG_LEVEL = 'WARNING'
        mock_config.LOG_FORMAT = '%(name)s - %(levelname)s - %(message)s'
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        setup_logging(log_file_path=None, level=None)
        assert root_logger.level == logging.WARNING

    @patch('src.utils.logging_config.config')
    def test_setup_logging_config_defaults_name(self, mock_config):
        mock_config.LOG_LEVEL = 'WARNING'
        mock_config.LOG_FORMAT = '%(name)s - %(levelname)s - %(message)s'
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        setup_logging(log_file_path=None, level=None)
        for handler in root_logger.handlers:
            if handler.formatter:
                format_string = handler.formatter._fmt
                assert '%(name)s' in format_string

    @patch('src.utils.logging_config.config')
    def test_setup_logging_config_defaults_levelname(self, mock_config):
        mock_config.LOG_LEVEL = 'WARNING'
        mock_config.LOG_FORMAT = '%(name)s - %(levelname)s - %(message)s'
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        setup_logging(log_file_path=None, level=None)
        for handler in root_logger.handlers:
            if handler.formatter:
                format_string = handler.formatter._fmt
                assert '%(levelname)s' in format_string

    @patch('src.utils.logging_config.config')
    def test_setup_logging_config_defaults_message(self, mock_config):
        mock_config.LOG_LEVEL = 'WARNING'
        mock_config.LOG_FORMAT = '%(name)s - %(levelname)s - %(message)s'
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        setup_logging(log_file_path=None, level=None)
        for handler in root_logger.handlers:
            if handler.formatter:
                format_string = handler.formatter._fmt
                assert '%(message)s' in format_string 