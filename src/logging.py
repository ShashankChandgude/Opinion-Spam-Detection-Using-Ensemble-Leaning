import logging
import sys

def configure_logging(log_file_path: str) -> None:
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    logging.root.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )
