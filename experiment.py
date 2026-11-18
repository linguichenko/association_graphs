from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_associations(path):
    """Читает таблицу SWOW с колонками cue, response, R1.Strength."""
    path = Path(path)
    with path.open(encoding="utf-8-sig") as file:
        separator = "\t" if "\t" in file.readline() else ","

    data = pd.read_csv(path, sep=separator)
    data = data[["cue", "response", "R1.Strength"]].copy()
    data.columns = ["cue", "response", "weight"]
    data["weight"] = pd.to_numeric(data["weight"], errors="coerce")
    data = data.dropna(subset=["cue", "response", "weight"])
    data = data[data["weight"] > 0].copy()

    data["cue"] = data["cue"].astype(str).str.strip().str.lower()
    data["response"] = data["response"].astype(str).str.strip().str.lower()
    data = data[(data["cue"] != "") & (data["response"] != "")]
    return data


def select_words(data, n_words=200):
    """Выбирает слова с наибольшей суммарной силой входящих ассоциаций, чтобы матрицы были поменьше.

    Рассматриваем только те слова, которые встречаются и как стимулы.
    """
    cues = set(data["cue"])
    answers = data[data["response"].isin(cues)]
    scores = answers.groupby("response")["weight"].sum()
    return scores.nlargest(n_words).index.tolist()


def build_matrix(data, words):
    """Строит A, где A[i, j] — сила ассоциации words[i] -> words[j]."""
    word_to_index = {word: i for i, word in enumerate(words)}
    n = len(words)
    matrix = np.zeros((n, n), dtype=float)

    subset = data[data["cue"].isin(words) & data["response"].isin(words)]
    for cue, response, weight in subset[["cue", "response", "weight"]].itertuples(
        index=False, name=None
    ):
        i = word_to_index[cue]
        j = word_to_index[response]
        matrix[i, j] += weight

    np.fill_diagonal(matrix, 0) 
    return matrix


def random_matrix(matrix, seed=42):
    """Случайно распределяет те же ненулевые веса по парам разных вершин.

    Сохраняет n, число дуг и список весов.
    """
    n = matrix.shape[0]
    weights = matrix[matrix > 0]
    rows, cols = np.where(~np.eye(n, dtype=bool))

    rng = np.random.default_rng(seed)
    positions = rng.choice(len(rows), size=len(weights), replace=False)
    result = np.zeros_like(matrix)
    result[rows[positions], cols[positions]] = rng.permutation(weights)
    return result


def spectrum(matrix):
    """Возвращает собственные значения и правые собственные векторы."""
    return np.linalg.eig(matrix)


def calculate_ipr(vectors):
    """IPR для каждого собственного вектора (столбца матрицы vectors)."""
    norms = np.linalg.norm(vectors, axis=0)
    normalized = vectors / norms
    return np.sum(np.abs(normalized) ** 4, axis=0)


def plot_spectra(values, random_values):
    """Два спектра на комплексной плоскости в одном масштабе."""
    limit = 1.1 * max(np.max(np.abs(values)), np.max(np.abs(random_values)))
    if limit == 0:
        limit = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, eigvals, title in zip(
        axes,
        (values, random_values),
        ("SWOW", "Случайный граф"),
    ):
        ax.scatter(eigvals.real, eigvals.imag, s=16, alpha=0.7)
        ax.set(title=title, xlabel="Re(λ)", ylabel="Im(λ)")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    return fig


def plot_ipr(values, vectors, random_values, random_vectors):
    """Сравнивает локализацию собственных векторов SWOW и случайного графа."""
    ipr = calculate_ipr(vectors)
    random_ipr = calculate_ipr(random_vectors)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(np.abs(values), ipr, s=18, alpha=0.65, label="SWOW")
    ax.scatter(
        np.abs(random_values), random_ipr,
        s=18, alpha=0.65, label="Случайный граф",
    )
    ax.axhline(1 / len(values), color="gray", linestyle="--", label="1/n")
    ax.set(xlabel="|λ|", ylabel="IPR")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def localized_words(values, vectors, words, n_vectors=3, n_words=8):
    """Показывает слова с наибольшими координатами локализованных векторов.

    Для простоты пропускаем почти нулевые собственные значения.
    """
    ipr = calculate_ipr(vectors)
    threshold = 0.2 * max(np.abs(values))
    candidates = np.flatnonzero(np.abs(values) > threshold)
    chosen = candidates[np.argsort(-ipr[candidates])[:n_vectors]]

    rows = []
    for k in chosen:
        top_indices = np.argsort(-np.abs(vectors[:, k]))[:n_words]
        rows.append({
            "lambda": f"{values[k]:.3f}",
            "IPR": round(float(ipr[k]), 4),
            "words": ", ".join(words[i] for i in top_indices),
        })
    return pd.DataFrame(rows, columns=["lambda", "IPR", "words"])


def load_lwow(path):
    """Читает обработанный FA_Llama3.csv (или FA_Mistral/FA_Haiku).

    В LWOW каждая строка соответствует одному повторению стимула,
    а R1 — первый ответ модели. Получаем ту же таблицу
    cue / response / weight, что и для SWOW.
    Вес — доля повторений, в которых появился этот ответ.
    """
    path = Path(path)

    with path.open(encoding="utf-8-sig") as file:
        separator = "\t" if "\t" in file.readline() else ","

    columns = pd.read_csv(path, sep=separator, nrows=0).columns.tolist()
    def find_column(names):
        for name in names:
            for original in columns:
                if original.strip().lower() == name:
                    return original
        return None

    cue_col = find_column(("cue", "stimulus", "cue_word"))
    response_col = find_column((
        "r1", "response1", "response_1", "response 1",
        "first_response", "r1.response", "r1_response",
    ))

    raw = pd.read_csv(path, sep=separator, usecols=[cue_col, response_col])
    raw = raw.rename(columns={cue_col: "cue", response_col: "response"})
    raw["cue"] = raw["cue"].astype("string").str.strip().str.lower()
    raw["response"] = raw["response"].astype("string").str.strip().str.lower()
    raw = raw.dropna(subset=["cue"])
    raw = raw[raw["cue"] != ""].copy()

    n_attempts = raw.groupby("cue").size()
    raw = raw.dropna(subset=["response"])
    raw = raw[raw["response"] != ""]

    counts = raw.groupby(["cue", "response"]).size().rename("count").reset_index()
    counts["weight"] = counts["count"] / counts["cue"].map(n_attempts)
    return counts[["cue", "response", "weight"]]


def plot_two_spectra(swow_values, lwow_values, model="Llama3"):
    """Спектры SWOW и LWOW в одинаковом масштабе."""
    limit = 1.1 * max(
        np.max(np.abs(swow_values)), np.max(np.abs(lwow_values)), 1e-6
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, values, name in zip(
        axes, (swow_values, lwow_values), ("SWOW (люди)", f"LWOW ({model})")
    ):
        ax.scatter(values.real, values.imag, s=16, alpha=0.7)
        ax.set_title(name)
        ax.set_xlabel("Re(λ)")
        ax.set_ylabel("Im(λ)")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_two_ipr(swow_values, swow_vectors, lwow_values, lwow_vectors,
                 model="Llama3"):
    """IPR правых собственных векторов людей и LLM."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(
        np.abs(swow_values), calculate_ipr(swow_vectors),
        s=18, alpha=0.65, label="SWOW",
    )
    ax.scatter(
        np.abs(lwow_values), calculate_ipr(lwow_vectors),
        s=18, alpha=0.65, label=f"LWOW ({model})",
    )
    ax.axhline(1 / len(swow_values), color="gray", linestyle="--", label="1/n")
    ax.set_xlabel("|λ|")
    ax.set_ylabel("IPR")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig
