import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, display_html, Markdown

import warnings
warnings.filterwarnings("ignore")

from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def print_md(text: str):
    display(Markdown(text))


def normalize_data(data: pd.DataFrame, from_idx: int=1, show: bool=False, TS=True) -> np.ndarray:
    """ нормализуем данные  """    
    X = data.iloc[:, from_idx:]
    
    
    if TS:
        # """ по длине ряда """
        scaled_data = np.squeeze(TimeSeriesScalerMinMax().fit_transform(X))
        show_data = scaled_data
    else: 
        # """ по колонке даты """
        scaled_data = StandardScaler().fit_transform(X.T).T
        show_data = scaled_data.T
    
    if show:
        plt.figure(figsize=(10, 7))
        plt.plot(show_data)
        plt.show()
        
    return scaled_data


def find_num_clusters(scaled_data: pd.DataFrame, in_metric="euclidean", in_range=range(2, 10)) -> None:
    """ поиск оптимального кол-ва кластеров - графически 
        @ in_metric = ["euclidean" | "dtw"]
    """
    print_md(f"<H4>Исследование возможного кол-ва кластеров</H4>")
    
    distortions = []
    silhouette = []
    K = in_range
    for k in tqdm(K):
        kmeanModel = TimeSeriesKMeans(n_clusters=k, metric=in_metric, n_jobs=-1, n_init=5, max_iter=50, random_state=42)
        kmeanModel.fit(scaled_data)
        distortions.append(kmeanModel.inertia_)
        silhouette.append(silhouette_score(scaled_data, kmeanModel.labels_, metric=in_metric))
        # silhouette_score считает насколько чисты класстеры
        
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(K, distortions, 'b-')
    ax2.plot(K, silhouette, 'r-')

    ax1.set_xlabel('# clusters')
    ax1.set_ylabel('Distortion', color='b')
    ax2.set_ylabel('Silhouette', color='r')

    plt.show()


def make_cl_model(scaled_data: pd.DataFrame, n_clusters, in_metric="euclidean", show=True) -> TimeSeriesKMeans:
    """ обучение модели на @n_clusters кластеров
        @ in_metric = ["euclidean" | "dtw"]
    """
    ts_kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric=in_metric, n_jobs=-1, max_iter=50, random_state=42)
    ts_kmeans.fit(scaled_data)

    if show:
        for cluster_number in range(n_clusters): # Построим усредненные ряды внутри каждого кластера
            plt.plot(ts_kmeans.cluster_centers_[cluster_number, :, 0].T, label=cluster_number)
        plt.title("Cluster centroids") 
        plt.legend()
        plt.show()
        
    return ts_kmeans


def _plot_cluster_by_model(data, col_cluster, sub_fig):
    ax = sub_fig.subplots(
        int(np.ceil(data.shape[0]/2)),
        2,
        # figsize=(15, 3*int(np.ceil(data.shape[0]/2)))
    )
    ax = ax.reshape(-1)
    
    for index, (_, row) in enumerate(data.iterrows()):
        ax[index].plot(row.iloc[1:-2])
        ax[index].set_title(f"{row.Ticker}\n{row[col_cluster]}")
        plt.xticks(rotation=45)
    
    sub_fig.set_facecolor('0.95')
    sub_fig.suptitle(col_cluster) #\nLeft side')
    # sub_fig.supxlabel('xlabel for subfigs[0]')
    
    
def plot_compare_cluster(data, label, num_samples = 12):
    NUM_SAMPLES = num_samples
    
    # данные разных моделей
    show_base = data[data['BASE_LABEL']==label]
    show_dtw = data[data['DTW_LABEL']==label]

    if show_base.shape[0] > NUM_SAMPLES:
                show_base = show_base.sample(NUM_SAMPLES, random_state=42)

    if show_dtw.shape[0] > NUM_SAMPLES:
                show_dtw = show_dtw.sample(NUM_SAMPLES, random_state=42)
    # отрисовка
    fig = plt.figure(layout="constrained",
                figsize=(15, 3*int(np.ceil(show_base.shape[0]/2)))
                )
    subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[1., 1.])
    _plot_cluster_by_model(show_base, 'BASE_LABEL', subfigs[0])
    _plot_cluster_by_model(show_dtw, 'DTW_LABEL', subfigs[1])
    
    # plt.tight_layout()
    plt.show()    
    
    
def show_compare_cluster(data, num_samples = 12):

    col_cluster = 'BASE_LABEL'
    for cluster in data[col_cluster].unique():
        print_md("<H5>===============================================</H5>")
        print_md(f"<H3>Cluster number: <b>{cluster}</b> ({num_samples} samples)</H3>")
        print_md("<H5>===============================================</H5>")

        plot_compare_cluster(data, cluster, num_samples = num_samples)
    

    
# def plot_cluster_tickers(current_cluster, col_cluster):
#     fig, ax = plt.subplots(
#         int(np.ceil(current_cluster.shape[0]/4)),
#         4,
#         figsize=(15, 3*int(np.ceil(current_cluster.shape[0]/4)))
#     )
#     fig.autofmt_xdate(rotation=45)
#     ax = ax.reshape(-1)

#     for index, (_, row) in enumerate(current_cluster.iterrows()):
#         ax[index].plot(row.iloc[1:-2])
#         ax[index].set_title(f"{row.Ticker}\n{row[col_cluster]}")
#         plt.xticks(rotation=45)

#     plt.tight_layout()
#     plt.show()
    

# def show_clusters_info(base_data: pd.DataFrame,  scaled_data: pd.DataFrame, model: TimeSeriesKMeans, col_cluster: str) -> pd.DataFrame:
#     """ Вывод результатов кластеризации временного ряда """
    
#     base_data[col_cluster] = model.predict(scaled_data)
#     print_md("<H3>Распределение по кластерам</H3>")
#     display(pd.DataFrame(base_data.groupby(col_cluster)['Ticker'].count()))
#     BTC_cluster = base_data.query('Ticker == "BTC" ')[col_cluster].values[0]
#     print_md(f'<H4>Биткойн принадлежит кластеру {BTC_cluster}</H4>')

#     for cluster in range(base_data[col_cluster].nunique()):
#         print_md("<H5>===============================================</H5>")
#         print_md(f"<H4>Cluster number: {cluster} (12 samples)</H4>")
#         print_md("<H5>===============================================</H5>")
        
#         show_data = base_data[base_data[col_cluster]==cluster]
#         if show_data.shape[0] > 12:
#             show_data = show_data.sample(12)
#         plot_cluster_tickers(show_data, col_cluster)
    

# def step_base(base_data: pd.DataFrame, in_metric: str='euclidean') -> pd.DataFrame:
#     """ Нормализация данных и исследование на кол-во кластеров"""
#     scaled = normalize_data(base_data, show=True)
#     print_md(f"<H4>Исследование возможного кол-ва кластеров</H4>")
#     find_num_clusters(scaled, in_metric=in_metric)
#     return scaled


# def step_model_info(base_data: pd.DataFrame,  scaled_data: pd.DataFrame, n_clusters: int, 
#     col_cluster: str, in_metric: str='euclidean') -> pd.DataFrame:
#     """ Обучение модели и вывод результатов кластеризации"""

#     model = make_cl_model(scaled_data, n_clusters, in_metric, show=True)
#     df_result = show_clusters_info(base_data,  scaled_data, model, col_cluster)
#     return df_result

