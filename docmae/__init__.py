import datetime
import logging
import os
import sys


def setup_logging(log_level, log_dir):
    """
    To set up logging
    :param log_level:
    :param log_dir:
    :return:
    """

    log_level = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }[log_level.upper()]
    root_logger = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    handler.setFormatter(formatter)
    root_logger.setLevel(log_level)
    root_logger.addHandler(handler)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_filename = datetime.datetime.now().strftime("%Y-%m-%d") + ".log"
        filehandler = logging.FileHandler(filename=os.path.join(log_dir, log_filename))
        filehandler.setFormatter(formatter)
        root_logger.addHandler(filehandler)
