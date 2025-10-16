import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from src.representations.word_embedder import WordEmbedder


def visualize_embeddings(model_name="glove-wiki-gigaword-50", method="pca"):
    print(f"🔹 Loading model: {model_name} ...")
    embedder = WordEmbedder(model_name)

    # Chọn một số từ tiêu biểu (có thể tùy chỉnh)
    words = [
        "king", "queen", "man", "woman", "prince", "princess",
        "apple", "banana", "orange", "fruit",
        "car", "truck", "vehicle", "road",
        "dog", "cat", "animal", "pet"
    ]

    # Lấy vector của từng từ
    vectors = []
    valid_words = []
    for word in words:
        vec = embedder.get_vector(word)
        if vec is not None:
            vectors.append(vec)
            valid_words.append(word)

    X = np.array(vectors)

    # Giảm chiều
    if method.lower() == "pca":
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, perplexity=5, random_state=42)

    reduced = reducer.fit_transform(X)

    # Vẽ scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1], color="steelblue")

    # Ghi nhãn từ
    for i, word in enumerate(valid_words):
        plt.annotate(word, (reduced[i, 0] + 0.02, reduced[i, 1] + 0.02))

    plt.title(f"Word Embeddings Visualization ({method.upper()})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Chạy thử với PCA
    # visualize_embeddings(method="pca")


     visualize_embeddings(method="tsne")
