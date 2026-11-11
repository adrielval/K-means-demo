Demonstração de K-means implementado do zero (sem bibliotecas externas de ML).
 
O que este script faz:
  1) gera dados sintéticos em 2D (três aglomerados)
  2) padroniza as colunas (média 0, desvio 1)
  3) usa o método do cotovelo para visualizar a escolha de k
  4) executa o K-means (k-means++ + várias inicializações)
  5) mostra os dados brutos e os clusters lado a lado
 
Uso no terminal:
    python kmeans_demo.py
    python kmeans_demo.py --k 4 --spread 2.0 --init aleatoria
 
Uso em um notebook:
    from kmeans_demo import kmeans
"""
 
import argparse
from typing import NamedTuple
 
import numpy as np
import matplotlib.pyplot as plt
 
 
# ---------------------------------------------------------------------------
# Estrutura do resultado
# ---------------------------------------------------------------------------
class ResultadoKMeans(NamedTuple):
    labels: np.ndarray     # (n,)   cluster de cada ponto
    centroids: np.ndarray  # (k, d) posição de cada centroide
    inertia: float         # soma das distâncias ao quadrado até o centroide
    n_iter: int            # iterações até convergir (ou até max_iter)
 
 
# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------
def distancias_quadradas(X, C):
    """
    Distância euclidiana AO QUADRADO entre cada ponto de X e cada centroide de C.
    Usa a expansão ||x - c||² = ||x||² - 2·x·c + ||c||², que evita criar
    um array (n, k, d) e gasta muito menos memória com n grande.
    Retorna uma matriz (n, k).
    """
    d2 = (X ** 2).sum(axis=1)[:, None] - 2 * X @ C.T + (C ** 2).sum(axis=1)[None, :]
    return np.maximum(d2, 0.0)  # corrige pequenos negativos por erro de arredondamento
 
 
def padronizar(X):
    """Deixa cada coluna com média 0 e desvio padrão 1. Retorna (X_pad, media, desvio)."""
    media = X.mean(axis=0)
    desvio = X.std(axis=0)
    desvio[desvio == 0] = 1.0  # evita divisão por zero em colunas constantes
    return (X - media) / desvio, media, desvio
 
 
def inicializar_centroides(X, k, metodo, rng):
    """
    'aleatoria': escolhe k pontos distintos ao acaso.
    'k-means++': escolhe o 1º ao acaso e cada próximo com probabilidade
                 proporcional à distância² até o centroide mais próximo,
                 o que espalha os centroides iniciais.
    """
    n = len(X)
    if metodo == "aleatoria":
        return X[rng.choice(n, size=k, replace=False)].astype(float)
 
    if metodo == "k-means++":
        centroids = np.empty((k, X.shape[1]))
        centroids[0] = X[rng.integers(n)]
        for j in range(1, k):
            d2_min = distancias_quadradas(X, centroids[:j]).min(axis=1)
            total = d2_min.sum()
            if total == 0:  # todos os pontos já coincidem com algum centroide
                centroids[j] = X[rng.integers(n)]
            else:
                centroids[j] = X[rng.choice(n, p=d2_min / total)]
        return centroids
 
    raise ValueError(f"Método de inicialização desconhecido: {metodo!r}")
 
 
# ---------------------------------------------------------------------------
# K-means
# ---------------------------------------------------------------------------
def _kmeans_uma_execucao(X, k, init, max_iter, tol, rng):
    """Uma execução completa do algoritmo de Lloyd."""
    n = len(X)
    centroids = inicializar_centroides(X, k, init, rng)
    n_iter = max_iter
 
    for it in range(1, max_iter + 1):
        # Etapa de atribuição: cada ponto vai para o centroide mais próximo
        d2 = distancias_quadradas(X, centroids)
        labels = d2.argmin(axis=1)
 
        # Etapa de atualização: cada centroide vai para a média dos seus pontos
        new_centroids = centroids.copy()
        dist_ao_proprio = d2[np.arange(n), labels]
        mais_distantes = np.argsort(dist_ao_proprio)[::-1]  # piores pontos primeiro
        usados = 0
        for c in range(k):
            pts = X[labels == c]
            if len(pts) > 0:
                new_centroids[c] = pts.mean(axis=0)
            else:
                # Cluster vazio: "ressuscita" o centroide no ponto mais mal
                # atendido, para ele não ficar parado para sempre
                new_centroids[c] = X[mais_distantes[usados]]
                usados += 1
 
        # Verifica a convergência pelo maior deslocamento de um centroide
        shift = np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1)).max()
        centroids = new_centroids
        if shift < tol:
            n_iter = it
            break
 
    # Recalcula os rótulos com os centroides FINAIS, para que labels,
    # centroides e inércia sejam sempre coerentes entre si
    d2 = distancias_quadradas(X, centroids)
    labels = d2.argmin(axis=1)
    inertia = float(d2[np.arange(n), labels].sum())
    return ResultadoKMeans(labels, centroids, inertia, n_iter)
 
 
def kmeans(X, k, init="k-means++", n_init=10, max_iter=300, tol=1e-4, random_state=None):
    """
    K-means com:
      - inicialização 'k-means++' (padrão) ou 'aleatoria'
      - n_init execuções independentes, mantendo a de menor inércia
      - distâncias ao quadrado (sem raiz desnecessária)
      - tratamento de clusters vazios
    Retorna um ResultadoKMeans(labels, centroids, inertia, n_iter).
    """
    X = np.asarray(X, dtype=float)
    if not 1 <= k <= len(X):
        raise ValueError(f"k deve estar entre 1 e {len(X)}, recebido {k}")
 
    rng = np.random.default_rng(random_state)
    melhor = None
    for _ in range(n_init):
        res = _kmeans_uma_execucao(X, k, init, max_iter, tol, rng)
        if melhor is None or res.inertia < melhor.inertia:
            melhor = res
    return melhor
 
 
# ---------------------------------------------------------------------------
# Dados e gráficos
# ---------------------------------------------------------------------------
def gerar_dados(n_per_cluster=150, spread=0.9, seed=42):
    """Três aglomerados gaussianos em 2D."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [5, 5], [0, 6]], dtype=float)
    return np.vstack([
        rng.normal(loc=c, scale=spread, size=(n_per_cluster, 2)) for c in centers
    ])
 
 
def grafico_cotovelo(X, k_max=8, random_state=0):
    """Plota a inércia para k = 1..k_max. O 'cotovelo' sugere um bom k."""
    ks = range(1, k_max + 1)
    inercias = [kmeans(X, k, random_state=random_state).inertia for k in ks]
 
    plt.figure(figsize=(6, 4))
    plt.plot(list(ks), inercias, marker="o")
    plt.title("Método do cotovelo")
    plt.xlabel("k (número de clusters)")
    plt.ylabel("Inércia")
    plt.xticks(list(ks))
    plt.grid(alpha=0.3)
    plt.tight_layout()
 
 
def grafico_antes_depois(X, resultado):
    """Dados brutos e dados agrupados lado a lado."""
    k = len(resultado.centroids)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
 
    ax1.scatter(X[:, 0], X[:, 1], s=16, alpha=0.8)
    ax1.set_title("Dados brutos (sem rótulos)")
 
    for c in range(k):
        pts = X[resultado.labels == c]
        ax2.scatter(pts[:, 0], pts[:, 1], s=16, alpha=0.8, label=f"Cluster {c}")
    ax2.scatter(resultado.centroids[:, 0], resultado.centroids[:, 1],
                s=160, marker="X", c="black", label="Centroides")
    ax2.set_title(f"K-means (k={k}) — Inércia: {resultado.inertia:.1f}")
    ax2.legend()
 
    for ax in (ax1, ax2):
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
    fig.tight_layout()
 
 
# ---------------------------------------------------------------------------
# Execução principal
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Demonstração de K-means do zero")
    parser.add_argument("--k", type=int, default=3, help="número de clusters")
    parser.add_argument("--spread", type=float, default=0.9, help="dispersão dos aglomerados")
    parser.add_argument("--init", choices=["k-means++", "aleatoria"], default="k-means++")
    parser.add_argument("--n-init", type=int, default=10, help="número de execuções")
    parser.add_argument("--seed", type=int, default=7, help="semente do K-means")
    parser.add_argument("--sem-graficos", action="store_true", help="só imprime os resultados")
    args = parser.parse_args()
 
    X = gerar_dados(spread=args.spread)
 
    # O K-means roda nos dados padronizados...
    X_pad, media, desvio = padronizar(X)
    res = kmeans(X_pad, args.k, init=args.init, n_init=args.n_init, random_state=args.seed)
 
    # ...e os centroides voltam para a escala original na hora de mostrar.
    # (A inércia exibida é a da escala padronizada.)
    res = res._replace(centroids=res.centroids * desvio + media)
 
    print("Centroides (escala original):\n", np.round(res.centroids, 3))
    print("Inércia (escala padronizada):", round(res.inertia, 2))
    print("Iterações até convergir:", res.n_iter)
    print("Pontos por cluster:", np.bincount(res.labels, minlength=args.k))
 
    if not args.sem_graficos:
        grafico_cotovelo(X_pad)
        grafico_antes_depois(X, res)
        plt.show()
 
 
if __name__ == "__main__":
    main()
 
Claude terminou a resposta
