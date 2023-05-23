import random
from typing import Union
from copywriting import get_logger
from copywriting import recovery_score, recovery_chinese_test, recovery_trueskill
from plots import plot_bar, plot_scatter, plot_humaneval_radar, plot_humaneval_bar, plot_ability_radar

logger = get_logger(__name__, 'INFO')

def recovery(name: str, 
             annotating_path: str, 
             dump_result_path:str, 
             annotated_path:str, 
             recovery_tasks:Union[list, str],
             save_fig_path: str,
             save_fig_name: str,
             plot_type: str,
             radar_type: str,
             seed: int):
    
    random.seed = seed

    if name == 'chinese_capability':
        recover_chinese = recovery_chinese_test(annotating_path, annotated_path, dump_result_path)
        plot_scatter(recover_chinese, save_fig_path=save_fig_path, save_name=save_fig_name)
        logger.info(recover_chinese)

    if name == 'rank_evaluation':
        for t in recovery_tasks:
            recovery_data = recovery_score(t, annotating_path, annotated_path, dump_result_path)
            plot_bar(recovery_data, save_fig_path=save_fig_path, save_name=save_fig_name)
            logger.info(recovery_data)

    if name == 'human_evaluation':
        if plot_type == 'bar':
            plot_humaneval_bar(annotated_path, save_fig_path, dump_result_path)
        else:
            assert plot_type == 'radar', f'The plot type {plot_type} is not supported!'
            if radar_type:
                plot_ability_radar(annotated_path, save_fig_path, radar_type)
            else:
                plot_humaneval_radar(annotated_path, save_fig_path)

    if name == 'trueskill_evaluation':
        if type(recovery_tasks) is str:
            task = recovery_tasks
        else:
            task = recovery_tasks[0]

        recover_trueskill = recovery_trueskill(task, annotating_path, annotated_path, dump_result_path)
        logger.info(recover_trueskill)