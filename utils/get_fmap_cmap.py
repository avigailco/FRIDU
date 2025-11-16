# Used colormap is adapted from the one defined in https://github.com/HTDerekLiu/SpecCoarsen_MATLAB

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def create_custom_colormap(type='default'):
    colormap_data = {
        'default': np.array([
            [103, 0, 31],
            [178, 24, 43],
            [214, 96, 77],
            [244, 165, 130],
            [253, 219, 199],
            [209, 229, 240],
            [146, 197, 222],
            [67, 147, 195],
            [33, 102, 172],
            [5, 48, 97]
        ]) / 255.0,

        'red': np.array([
            [255, 247, 236],
            [254, 232, 200],
            [253, 212, 158],
            [253, 187, 132],
            [252, 141, 89],
            [239, 101, 72],
            [215, 48, 31],
            [179, 0, 0],
            [127, 0, 0]
        ]) / 255.0,

        'blue': np.array([
            [255, 247, 251],
            [236, 231, 242],
            [208, 209, 230],
            [166, 189, 219],
            [116, 169, 207],
            [54, 144, 192],
            [5, 112, 176],
            [4, 90, 141],
            [2, 56, 88]
        ]) / 255.0,

        'heat': np.array([
            [255, 247, 236],
            [255, 242, 224],
            [254, 237, 212],
            [254, 232, 200],
            [253, 222, 179],
            [253, 212, 158],
            [253, 200, 145],
            [253, 187, 132],
            [252, 141, 89],
            [239, 101, 72],
            [215, 48, 31],
            [179, 0, 0],
            [127, 0, 0]
        ]) / 255.0,

        'gray': np.linspace(255, 0, 11).reshape(-1, 1).repeat(3, axis=1) / 255.0
    }

    if type not in colormap_data:
        raise ValueError(f"Unknown colormap type: {type}")

    colors = colormap_data[type]
    cmap = LinearSegmentedColormap.from_list(f"custom_{type}", colors, N=500)

    return cmap


if __name__ == '__main__':
    # Example usage and visualization
    colormap_name = 'default'  # Change to 'red', 'blue', 'heat', 'gray' for other colormaps
    cmap = create_custom_colormap(colormap_name)

    # Display the colormap
    fig, ax = plt.subplots(figsize=(6, 1))
    gradient = np.linspace(0, 1, 500).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.set_title(f"Custom Colormap: {colormap_name}")
    ax.set_axis_off()
    plt.show()
