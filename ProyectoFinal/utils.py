import matplotlib.pyplot as plt
from matplotlib import rcParams

def plot_loss(losses, title=None):
    rcParams['font.style'] = 'normal'
    rcParams['font.size'] = 12
    rcParams['font.weight'] = 'normal'
    plt.figure()
    plt.plot(losses)
    #plt.ylim((1055476.62*0.96, 1265412.12*1.2))
    plt.grid(True)
    plt.xlabel('Iteración')
    plt.ylabel(r'loss$={||X-DA||_F^2}$')
    if title is not None: plt.title(title)