import matplotlib.pyplot as plt


def plot_map(map_s, depth, num_train, num_val):
    plt.figure(figsize=(7, 10))
    plt.scatter(x=map_s[:, 0], y=map_s[:, 1], c=map_s[:, 2], s=1, cmap='rainbow')
    plt.title(f'{depth} м. Number of training and validation points: {num_train + num_val}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('scaled')
    plt.tight_layout()
    plt.colorbar(shrink=0.25, label='Vs', orientation='horizontal')
    return plt
