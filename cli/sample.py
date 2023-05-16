import random
import argparse
from copywriting import get_logger
from copywriting import sample_instance, sample_chinese_testing

logger = get_logger(__name__, 'INFO')

def sample(name :str, 
           original_file_path :str, 
           annotating_path :str, 
           dump_recovery_path :str, 
           single_sample_size :int, 
           sample_num :int, 
           sample_llm_num :int, 
           llm_candidations :list, 
           evaluation_tasks :list,
           seed: int):

    random.seed = seed

    if name == 'chinese_capability':
        sample_stistics = sample_chinese_testing(single_sample_size, 
                                                sample_num, 
                                                sample_llm_num, 
                                                original_file_path, 
                                                evaluation_tasks, 
                                                llm_candidations, 
                                                annotating_path, 
                                                dump_recovery_path)

        logger.info(sample_stistics)

    if name == 'rank_evaluation':
        for t in evaluation_tasks:
            sample_instance(sample_num, 
                            original_file_path, 
                            t, 
                            llm_candidations, 
                            annotating_path, 
                            dump_recovery_path)