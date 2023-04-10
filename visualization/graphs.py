from matplotlib import pyplot as plt
import PIL

def mk_pil_graph(dims, data, title, xlabel, ylabel):
    plt.tight_layout(rect=(0.1, 0.0, 1, 1))
    fig = plt.figure(figsize=(dims[0] / 100, dims[1] / 100))
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig.canvas.draw()
    img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close()
    return img

def mk_pil_table(dims, data, title, xlabel=None, ylabel=None):
    plt.tight_layout(rect=(0.1, 0.0, 1, 1))
    fig = plt.figure(figsize=(dims[0] / 100, dims[1] / 100))
    plt.table(cellText=data, loc='center')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.title(title)
    fig.canvas.draw()
    img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close()
    return img