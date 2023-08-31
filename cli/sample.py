import random
from typing import Union
from analysis import get_logger
from analysis import sample_instance, sample_chinese_testing, sample_trueskill

logger = get_logger(__name__, 'INFO')

def sample(name :str, 
           original_file_path :str, 
           annotating_path :str, 
           dump_recovery_path :str, 
           single_sample_size :int, 
           sample_num :int, 
           sample_llm_num :int, 
           llm_candidations :list, 
           evaluation_tasks :Union[list, str],
           match_plan :list,
           seed: int):

    random.seed = seed

    if name == 'chinese_capability':
        sample_stistics = sample_chinese_testing(sample_num,
                                                single_sample_size, 
                                                sample_llm_num, 
                                                original_file_path, 
                                                evaluation_tasks, 
                                                llm_candidations, 
                                                annotating_path, 
                                                dump_recovery_path)

        logger.info(sample_stistics)

    if name == 'rank_evaluation':
        for task in evaluation_tasks:
            sample_instance(sample_num, 
                            original_file_path, 
                            task, 
                            llm_candidations, 
                            annotating_path, 
                            dump_recovery_path)
            
    if name == 'trueskill_evaluation':
        if type(evaluation_tasks) is str:
            task = evaluation_tasks
        else:
            task = evaluation_tasks[0]
            
        sample_trueskill(match_plan, 
                         task, 
                         sample_num, 
                         original_file_path, 
                         annotating_path, 
                         dump_recovery_path)