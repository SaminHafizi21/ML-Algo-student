"""Genereert de figuren voor les06 (DBSCAN) slides, gebaseerd op notebooks/week06dbscan.ipynb.

Volgt de stijl van generate_les05_figs.py.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import KMeans, DBSCAN

OUT = Path(__file__).parent / "images"
OUT.mkdir(exist_ok=True)


def save_fig(fig_id):
    plt.tight_layout()
    plt.savefig(OUT / f"{fig_id}.png", format="png", dpi=200)
    plt.close()


def plot_moons(X, y=None, s=20):
    if y is not None:
        plt.scatter(X[:, 0], X[:, 1], c=y, s=s, alpha=0.8)
    else:
        plt.plot(X[:, 0], X[:, 1], "k.", markersize=3)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$", rotation=0)


def plot_dbscan(dbscan, X, size=100, show_xlabels=True, show_ylabels=True):
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    anomalies_mask = dbscan.labels_ == -1
    non_core_mask = ~(core_mask | anomalies_mask)

    cores = dbscan.components_
    anomalies = X[anomalies_mask]
    non_cores = X[non_core_mask]

    plt.scatter(cores[:, 0], cores[:, 1],
                c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
    plt.scatter(non_cores[:, 0], non_cores[:, 1],
                c=dbscan.labels_[non_core_mask], marker=".", s=40)
    plt.scatter(anomalies[:, 0], anomalies[:, 1], c="r", marker="x", s=100)
    if show_xlabels:
        plt.xlabel("$x_1$")
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", rotation=0)
    else:
        plt.tick_params(labelleft=False)
    plt.title(f"eps={dbscan.eps:.2f}, min_samples={dbscan.min_samples}")
    plt.grid()
    plt.gca().set_axisbelow(True)


X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)

# 1. moons dataset
plt.figure(figsize=(8, 4))
plot_moons(X, y)
plt.gca().set_axisbelow(True)
plt.grid()
save_fig("les06_moons_plot")

# 2. KMeans op moons: werkt niet goed
kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X)
plt.figure(figsize=(8, 4))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=20, alpha=0.8)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c="red", marker="x", label="Clustercenters")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$", rotation=0)
plt.grid()
plt.gca().set_axisbelow(True)
plt.legend()
save_fig("les06_kmeans_moons_plot")

# 3. DBSCAN op moons: werkt
dbscan_05 = DBSCAN(eps=0.05, min_samples=5)
dbscan_05.fit(X)
plt.figure(figsize=(8, 4))
plot_dbscan(dbscan_05, X, size=80)
save_fig("les06_dbscan_moons_plot")

# 4. eps demo: te kleine versus te grote eps
dbscan_big = DBSCAN(eps=0.3)
dbscan_big.fit(X)
plt.figure(figsize=(9, 3.2))
plt.subplot(121)
plot_dbscan(dbscan_05, X, size=100)
plt.subplot(122)
plot_dbscan(dbscan_big, X, size=100, show_ylabels=False)
save_fig("les06_dbscan_eps_plot")

# 5. schets werking: core / border / anomalie met eps-cirkels
rng = np.random.RandomState(42)
center = np.array([2, 2])
points = center + rng.randn(60, 2) * 0.5
# een paar 'losse' punten als anomalie
anom = np.array([[0.2, 0.4], [3.8, 0.6], [0.6, 3.9], [4.0, 3.6]])
eps = 0.9
min_samples = 5
plotable = np.vstack([points, anom])

# simpele core/border/anon-bepaling op basis van buurten
def nb_in_eps(i):
    return np.sum(np.linalg.norm(plotable - plotable[i], axis=1) <= eps) - 1

is_core = np.array([nb_in_eps(i) >= min_samples for i in range(len(plotable))])
is_anom = ~is_core & np.array([nb_in_eps(i) == 0 for i in range(len(plotable))])
is_border = ~is_core & ~is_anom

plt.figure(figsize=(6, 5))
# eps-cirkels rond enkele core punten
for i in np.where(is_core)[0][:6]:
    circle = plt.Circle(plotable[i], eps, color="skyblue", alpha=0.25, lw=1)
    plt.gca().add_patch(circle)
plt.scatter(plotable[is_core, 0], plotable[is_core, 1],
            c="tab:blue", marker="o", s=90, label="Core point")
plt.scatter(plotable[is_border, 0], plotable[is_border, 1],
            c="tab:green", marker=".", s=80, label="Border point")
plt.scatter(plotable[is_anom, 0], plotable[is_anom, 1],
            c="red", marker="x", s=120, label="Anomalie")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$", rotation=0)
plt.grid()
plt.gca().set_axisbelow(True)
plt.legend(loc="upper left")
plt.title("core / border / anomalie")
save_fig("les06_dbscan_schets_plot")

# 6. DBSCAN vs HDBSCAN bij verschillende dichtheden (moeite met eps-keuze)
# -> genereren we zonder hdbscan (niet noodzakelijk geïnstalleerd), via DBSCAN per eps
centers = [[-0.85, -0.85], [-0.85, 0.85], [3, 3], [3, -3]]
Xv, yv = make_blobs(n_samples=750, centers=centers,
                    cluster_std=[0.2, 0.35, 1.35, 1.35], random_state=0)

plt.figure(figsize=(9, 3.2))
for idx, epsv in enumerate([0.7, 0.3]):
    plt.subplot(1, 2, idx + 1)
    dbs = DBSCAN(eps=epsv).fit(Xv)
    labels = dbs.labels_
    plt.scatter(Xv[:, 0], Xv[:, 1], c=labels, s=12, cmap="Paired")
    plt.scatter(Xv[labels == -1, 0], Xv[labels == -1, 1],
                c="black", marker="x", s=40)
    plt.title(f"DBSCAN eps={epsv}")
    plt.xticks(()); plt.yticks(())
save_fig("les06_dbscan_gevoeligheid_plot")

# 7. BIRCH op blobs
dataset, clusters = make_blobs(n_samples=600, centers=8,
                               cluster_std=0.75, random_state=0)
from sklearn.cluster import Birch
model = Birch(branching_factor=50, n_clusters=None, threshold=1.5)
pred = model.fit_predict(dataset)
plt.figure(figsize=(6, 5))
plt.scatter(dataset[:, 0], dataset[:, 1], c=pred, cmap="rainbow",
            alpha=0.8, edgecolors="k", linewidths=0.3)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$", rotation=0)
plt.title("BIRCH")
plt.grid()
plt.gca().set_axisbelow(True)
save_fig("les06_birch_plot")

print("Klaar:", sorted(p.name for p in OUT.glob("les06_*.png")))
