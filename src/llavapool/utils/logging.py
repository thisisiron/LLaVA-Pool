import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .constants import RUNNING_LOG


_thread_lock = threading.RLock()
_default_handler: Optional["logging.Handler"] = None
_default_log_level: "logging._Level" = logging.INFO


_formatter = logging.Formatter(
    fmt="%(asctime)s | llavapool | %(levelname)-5s | %(filename)s:%(lineno)d:%(funcName)s >>  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class LoggerHandler(logging.Handler):
    r"""
    Redirects the logging output to the logging file for LLaMA Board.
    """

    def __init__(self, output_dir: str) -> None:
        super().__init__()
        formatter = _formatter
        self.setLevel(logging.INFO)
        self.setFormatter(formatter)

        os.makedirs(output_dir, exist_ok=True)
        self.running_log = os.path.join(output_dir, RUNNING_LOG)
        if os.path.exists(self.running_log):
            os.remove(self.running_log)

        self.thread_pool = ThreadPoolExecutor(max_workers=1)

    def _write_log(self, log_entry: str) -> None:
        with open(self.running_log, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n\n")

    def emit(self, record) -> None:
        if record.name == "httpx":
            return

        log_entry = self.format(record)
        self.thread_pool.submit(self._write_log, log_entry)

    def close(self) -> None:
        self.thread_pool.shutdown(wait=True)
        return super().close()


def _get_default_logging_level() -> "logging._Level":
    r"""
    Returns the default logging level.
    """
    env_level_str = os.environ.get("LLAVAPOOL_VERBOSITY", None)
    if env_level_str:
        if env_level_str.upper() in logging._nameToLevel:
            return logging._nameToLevel[env_level_str.upper()]
        else:
            raise ValueError("Unknown logging level: {}.".format(env_level_str))

    return _default_log_level


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> "logging.Logger":
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:
    r"""
    Configures root logger using a stdout stream handler with an explicit format.
    """
    global _default_handler

    with _thread_lock:
        if _default_handler:
            return

        formatter = _formatter
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.setFormatter(formatter)
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())
        library_root_logger.propagate = False


def get_logger(name: Optional[str] = None) -> "logging.Logger":
    r"""
    Returns a logger with the specified name. It it not supposed to be accessed externally.
    """
    if name is None:
        name = _get_library_name()

    _configure_library_root_logger()
    return logging.getLogger(name)
