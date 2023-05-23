from plots import plot_hotmap, plot_gaussian, plot_dynamic_gif

def plot(type: str, data_file: str, save_fig_path: str, save_fig_name: str):
    if type == 'hotmap':
        plot_hotmap(data_file, save_fig_path, save_fig_name)
    
    if type == 'gaussian':
        plot_gaussian(data_file, save_fig_path, save_fig_name)

    if type == 'dynamic':
        plot_dynamic_gif(data_file, save_fig_path, save_fig_name)