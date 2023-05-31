from plots import plot_hotmap, plot_gaussian, plot_dynamic_gif, plot_vedio, plot_bar, plot_ability_radar

def plot(type: str, data_file: str, save_fig_path: str, save_fig_name: str):
    if type == 'bar':
        plot_bar(data_file, save_fig_path, save_fig_name)

    if type == 'hotmap':
        plot_hotmap(data_file, save_fig_path, save_fig_name)
    
    if type == 'gaussian':
        plot_gaussian(data_file, save_fig_path, save_fig_name)

    if type == 'dynamic':
        plot_dynamic_gif(data_file, save_fig_path, save_fig_name)

    if type == 'video':
        plot_vedio(data_file, save_fig_path, save_fig_name)

    if type == 'radar':
        plot_ability_radar(data_file, save_fig_path, save_fig_name)