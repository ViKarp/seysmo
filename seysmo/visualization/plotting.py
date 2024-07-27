import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from matplotlib.gridspec import GridSpec


def plot_map(map_s_true, map_s_pred, depth, num_train, num_val, num_test):
    # plt.figure(figsize=(7, 10))
    # plt.scatter(x=map_s[:, 0], y=map_s[:, 1], c=map_s[:, 2], s=1, cmap='rainbow')
    # plt.title(f'{depth} м. Number of training and validation points: {num_train + num_val}')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.axis('scaled')
    # plt.tight_layout()
    # plt.colorbar(shrink=0.25, label='Vs', orientation='horizontal')
    # Определяем границы цветовой шкалы на основе объединенных данных
    vmin = min(map_s_true[:, 2].min(), map_s_pred[:, 2].min())
    vmax = max(map_s_true[:, 2].max(), map_s_pred[:, 2].max())

    # Создаем фигуру и сетку для двух подграфиков
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.05)

    # Первый подграфик
    ax1 = fig.add_subplot(gs[0, 0])
    sc1 = ax1.scatter(x=map_s_true[:, 0], y=map_s_true[:, 1], c=map_s_true[:, 2], s=1, cmap='rainbow', vmin=vmin,
                      vmax=vmax)
    ax1.set_title(f'True map')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.axis('scaled')

    # Второй подграфик
    ax2 = fig.add_subplot(gs[0, 1])
    sc2 = ax2.scatter(x=map_s_pred[:, 0], y=map_s_pred[:, 1], c=map_s_pred[:, 2], s=1, cmap='rainbow', vmin=vmin,
                      vmax=vmax)
    ax2.set_title('Predict Map')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.axis('scaled')

    # Добавляем общий colorbar для обеих карт
    cbar = fig.colorbar(sc2, ax=[ax1, ax2], shrink=0.75, label='Vs', orientation='horizontal')
    plt.tight_layout()
    fig.suptitle(
        f'{depth} м. Number of training and validation points: {num_train + num_val} of all: {num_train + num_val + num_test}',
        fontsize=16)
    return fig
