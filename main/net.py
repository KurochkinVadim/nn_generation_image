# Importing libraries
from collections import Counter
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def init_normal(m):
    """
    Initialization of network weights with normal distribution.

    Parameters
    __________
    m : NN
        Network for initialization

    """
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight)


class NN(nn.Module):
    """
    A class for representing a network.

    Methods
    _______
    forward(x):
        Building an architecture
    """

    def __init__(self, activation=nn.Tanh, num_neurons=16, num_layers=7):
        """
        Initializing layers

        Parameters
        __________
        activation : optional
            Activating layers
        num_neurons : int
            The number of neurons on the linear layer
        num_layers : int
            Number of linear layers in the network

        Returns
        _______
        None

        """
        super(NN, self).__init__()
        layers = [nn.Linear(2, num_neurons, bias=True), activation()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(num_neurons, num_neurons, bias=False), activation()]
        layers += [nn.Linear(num_neurons, 3, bias=False)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def controlColors(colors_map, size_y, size_x, main_color, second_color=None, distance=0):
    """
    Color control at the output of the neural network

    Parameters
    __________
    colors_map : Tensor
        Model Output
    size_y : int
        The number of values in the Y coordinate
    size_x : int
        The number of values in the X coordinate
    main_color : str
        Main color
    second_color : str
        Second color
    distance : int
        Conditional distance between colors on the palette

    Returns
    _______
    colors_map : Tensor
        Model output after applying custom activation with color control

    """

    # Initialization of values for sigmoids of certain colors
    palettes = {
        'Red': (0.75, 0, 0),
        'Pink': (0.75, 0, 0.75),
        'Blue': (0, 0, 0.75),
        'Cyan': (0, 0.75, 0.75),
        'Green': (0, 0.75, 0),
        'Yellow': (0.75, 0.75, 0),
    }

    # Control of values provided that the colors are nearby
    if distance == 1 and second_color in ['Red', 'Blue', 'Green']:
        main_color, second_color = second_color, main_color
    rgb = [[0.25, palettes[main_color][0]], [0.25, palettes[main_color][1]], [0.25, palettes[main_color][2]]]
    if distance == 1:
        if second_color in ['Red', 'Blue', 'Green']:
            main_color, second_color = second_color, main_color
        for i in range(3):
            if palettes[main_color][i] != palettes[second_color][i]:
                rgb[i][0] = 1

    # Control of values provided that the colors are through one
    if distance == 2:
        if second_color in ['Red', 'Blue', 'Green']:
            for i in range(3):
                if palettes[main_color][i] != 0:
                    rgb[i][0] = 0.75
                    rgb[i][1] = 0.25
                if palettes[second_color][i] != 0:
                    rgb[i][0] = 0.75
                    rgb[i][1] = 0.25
        else:
            for i in range(3):
                if palettes[main_color][i] != palettes[second_color][i]:
                    rgb[i][0] = 1
                    rgb[i][1] = 0

    # Applying sigmoids to model output values
    for i in range(size_y):
        for j in range(size_x):
            colors_map[i][j][0] = rgb[0][0] / (1 + np.exp(-colors_map[i][j][0])) + rgb[0][1]
            colors_map[i][j][1] = rgb[1][0] / (1 + np.exp(-colors_map[i][j][1])) + rgb[1][1]
            colors_map[i][j][2] = rgb[2][0] / (1 + np.exp(-colors_map[i][j][2])) + rgb[2][1]
    return colors_map


def main_colors(tags, groups):
    """
    The function determines the primary and secondary color based on the input tags

    Parameters
    __________
    tags : list
        Tags at the entrance
    groups : dict
        Dictionary of matching tags and colors

    Returns
    _______
    main_color : str
        Main color
    second_color : str
        Second color
    distance : int
        The distance between colors on the palette

    """

    colors = ['Red', 'Pink', 'Blue', 'Cyan', 'Green', 'Yellow']
    new_tags = []
    for tag in tags:
        for color in groups:
            if tag in groups[color]:
                new_tags.append(color)
    counter = list(Counter(new_tags).most_common(6))
    main_color = counter[0][0]
    if len(counter) == 1:
        return main_color, main_color, 0
    for i in range(1, len(counter)):
        second_color = counter[i][0]
        distance = abs(colors.index(main_color) - colors.index(second_color))
        if distance == 2 or distance == 4:
            return main_color, second_color, 2
        if distance == 1 or distance == 5:
            return main_color, second_color, 1
    return main_color, main_color, 0


def gen_new_image(size_y, size_x, main_color, second_color, distance, save=True):
    """
    The main function. Generates an image based on the resolution, primary colors and the distance between them

    Parameters
    __________
    size_y : int
        The number of values in the Y coordinate
    size_x : int
        The number of values in the X coordinate
    main_color : str
        Main color
    second_color : str
        Second color
    distance : int
        The distance between colors on the palette
    save : bool
        if the parameter is set to "True", the image will be saved to the results directory

    Returns
    _______
    net : NN
        The model initialized by weights that was applied to generate
    colors : Tensor
        Image in the form of a tensor

    """
    # Defining and initializing the model
    net = NN()
    net.apply(init_normal)

    # Launching the model
    colors = run_net(net, size_y, size_x)
    print("The main drawing has been generated...")

    # Application of sigmoids for color control
    controlColors(colors, size_y, size_x, main_color, second_color, distance)
    print("The final image is ready...")

    # Rendering an image
    plot_colors(colors)

    # Saving an image
    if save is True:
        save_colors(colors)
        print("Image saved...")
    return net, colors


def run_net(net, size_y=128, size_x=128):
    """
    The function starts the network and generates the initial tensor without using the activation function at the end

    Parameters
    __________
    net : NN
        Model
    size_y : int
        The number of values in the Y coordinate
    size_x : int
        The number of values in the X coordinate

    Returns
    _______
    result : Tensor
        Returns the output tensor of the model

    """
    x = np.arange(0, size_x, 1)
    y = np.arange(0, size_y, 1)
    colors = np.zeros((size_y, size_x, 2))
    for i in y:
        for j in x:
            colors[i][j] = np.array([float(i) / size_x - 0.5, float(j) / size_y - 0.5])
    colors = colors.reshape(size_x * size_y, 2)
    img = net(torch.tensor(colors).type(torch.FloatTensor)).detach().numpy()
    result = img.reshape(size_y, size_x, 3)
    return result


def plot_colors(colors, fig_size=12):
    """
    The function renders the image

    Parameters
    __________
    colors : Tensor
        Image in the form of a tensor
    fig_size : int
        Image size

    """
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(colors, interpolation='nearest', vmin=0, vmax=1)


def save_colors(colors, name='default'):
    """
    The function saves the image

    Parameters
    __________
    colors : Tensor
        Image in the form of a tensor
    name : str
        Name of image

    """
    plt.imsave("results/" + name + ".png", colors)


def run_plot_save(net, size_x, size_y, main_color, second_color, distance, fig_size=8):
    """
    The function generates, saves and renders a new image without initializing a new network

    Parameters
    __________
    net : NN
        Model
    size_y : int
        The number of values in the Y coordinate
    size_x : int
        The number of values in the X coordinate
    main_color : str
        Main color
    second_color : str
        Second color
    distance : int
        The distance between colors on the palette
    fig_size : int
        Image size

    """
    colors = run_net(net, size_x, size_y)
    print("The main drawing has been generated...")
    controlColors(colors, size_y, size_x, main_color, second_color, distance)
    print("The final image is ready...")
    plot_colors(colors, fig_size)
    save_colors(colors)


def startGenerate(size_y, size_x, tags, groups, save=True):
    """
    The function determines colors by tags and generates an image

    Parameters
    __________
    size_y : int
        The number of values in the Y coordinate
    size_x : int
        The number of values in the X coordinate
    tags : list
        List of incoming tags
    groups : dist
        Dictionary of matching tags and colors

    Returns
    _______
    net : NN
        The model initialized by weights that was applied to generate
    colors : Tensor
        Image in the form of a tensor
    """

    # Definition of primary colors
    main_color, second_color, distance = main_colors(tags, groups)
    print("The primary colors are defined...")

    # Launching the image generation function
    net, colors = gen_new_image(size_y, size_x, main_color, second_color, distance, save)
    return net, colors
