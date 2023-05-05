import logging

def get_logger(name, level='DEBUG'):
    logging_level = eval(f'logging.{level}')
    logger = logging.getLogger(name)
    logger.setLevel(logging_level)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging_level)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    return logger

from .visit_api import visit_llm_api
