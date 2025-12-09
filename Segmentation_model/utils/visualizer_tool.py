# -- coding: utf-8 --
from visdom import Visdom
import numpy as np

class Visualizer():
    def __init__(self, env="default", **kwargs):  # **kwargs is necessary to flexibly add various parameters such as opts
        self.vis = Visdom(env=env, **kwargs)  # Initialization can be directly copied or assigned through parameters
        self.index = {}  # Create an empty dictionary to facilitate the calling and modification of values for different windows

    # The plot method is used for relay races, dynamically continuing to draw remaining points based on the static drawing of plot_line
    def plot(self, win, y, con_point, x=None, **kwargs):
        if x is not None:
            x = x
        else:
            x = self.index.get(win, con_point)  # ----dict.update(key,value), similar to list.append() in lists
        # The get method in dict: if the key (win) exists in the dict, return its value; otherwise, return con_point
        self.vis.line(Y=np.array([y]), X=np.array([x]), win=str(win), update=None if x == 0 else "append", **kwargs)
        self.index[win] = x + 1  # Increment the value corresponding to the current window

    # Used for static drawing, where values are taken from lists and drawn all at once; if win is consistent with plot, it can dynamically continue drawing
    def plot_line(self, win, y, **kwargs):
        self.vis.line(win=win, X=np.linspace(1, len(y), len(y)), Y=y, **kwargs)

    def img(self, name, img_, **kwargs):
        # images can draw images in BCHW format, with the number of images being the count of B, and the number of images per row can be specified
        self.vis.images(img_, win=str(name),  # Window name
                        opts=dict(title=name),  # Image title
                        **kwargs)
        
    def plot_pred_contrast(self, pred, label, image):
        self.img(name="image", img_=image)
        self.img(name="pred", img_=pred)
        self.img(name="label", img_=label)
    
    def plot_entropy(self, H):
        self.img(name="pred_entropy", img_=H)

    def plot_metrics_total(self, metrics_dict):
        """
        Function: Plot the total metrics curve; refer to the log file for specific category metrics!
        """
        # Only need to: 1. Remove data from metrics_dict that do not need to be displayed; 2. Add the calculation method for get_single_indicator; the rest remains unchanged, metrics are only added, not reduced; 3. Slightly modify the function return value
        for metric, values in metrics_dict.items():
            self.plot(win=metric, y=values['total'][-1], opts=dict(title=metric, xlabel="Epoch", ylabel=metric), con_point=len(values['total']))

    def plot_metrics_single(self, metrics_dict):
        """
        Function: Plot the line chart for each category's metrics
        """
        # Only need to: 1. Remove data from metrics_dict that do not need to be displayed; 2. Add the calculation method for get_single_indicator; the rest remains unchanged, metrics are only added, not reduced; 3. Slightly modify the function return value
        for metric, values in metrics_dict.items():
            for task, task_values in values["total"][-1].items():
                self.plot(win=f"{metric}_{task}", y=task_values, opts=dict(title=f"{metric} - {task}", xlabel="Epoch", ylabel=metric), con_point=len(values["total"]))

    def plot_metrics_Test(self, metrics_dict, vis_name, number):
        """
        Function: Plot the total metrics curve; refer to the log file for specific category metrics!
        """
        # Only need to: 1. Remove data from metrics_dict that do not need to be displayed; 2. Add the calculation method for get_single_indicator; the rest remains unchanged, metrics are only added, not reduced; 3. Slightly modify the function return value
        for metric, values in metrics_dict.items():
            self.plot(win=metric, y=values['total'][-1], opts=dict(title=metric, xlabel="Epoch", ylabel=metric), con_point=number, name=vis_name)
