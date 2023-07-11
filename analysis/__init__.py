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

from .visit_api import visit_llm_api, revisit_llm_api, visit_llm
from .sample_data import sample_instance, \
recovery_score, \
sample_chinese_testing, \
recovery_chinese_test, \
sample_trueskill, \
recovery_trueskill

from .process_data import human_evaluation_reader, \
human_annotation_reader, \
trueskill_hotmap_reader, \
trueskill_gaussian_reader, \
trustable_humaneval_creation

from .plot import plot_scatter, \
plot_humaneval_radar, \
plot_ability_radar, \
plot_hotmap, \
plot_gaussian, \
plot_dynamic_gif, \
plot_video, \
plot_bar, \
plot_icc
