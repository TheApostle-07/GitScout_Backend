import logging

def setup_logger():
    logger = logging.getLogger("gitscout_logger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(levelname)s] %(asctime)s: %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger