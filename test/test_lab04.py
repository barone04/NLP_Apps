import numpy as np
from src.representations.word_embedder import WordEmbedder


def main():
    print("🔹 Initializing WordEmbedder with 'glove-wiki-gigaword-50'...")
    embedder = WordEmbedder("glove-wiki-gigaword-50")

    # 1️⃣ Get the vector for the word ‘king’
    print("\n=== 1. Vector for 'king' ===")
    vec_king = embedder.get_vector("king")
    if vec_king is not None:
        print(f"Vector shape: {vec_king.shape}")
        print(vec_king)
    else:
        print("Word 'king' not found in vocabulary.")

    # 2️⃣ Similarity between ‘king’ and ‘queen’, and between ‘king’ and ‘man’
    print("\n=== 2. Similarities ===")
    sim_kq = embedder.get_similarity("king", "queen")
    sim_km = embedder.get_similarity("king", "man")
    print(f"Similarity(king, queen): {sim_kq:.4f}" if sim_kq is not None else "One word missing.")
    print(f"Similarity(king, man):   {sim_km:.4f}" if sim_km is not None else "One word missing.")

    # 3️⃣ 10 most similar words to ‘computer’
    print("\n=== 3. Top 10 words similar to 'computer' ===")
    similar_computer = embedder.get_most_similar("computer", top_n=10)
    if similar_computer:
        for word, score in similar_computer:
            print(f"{word:<15} {score:.4f}")
    else:
        print("Word 'computer' not found in vocabulary.")

    # 4️⃣ Embed the sentence “The queen rules the country.”
    print("\n=== 4. Document embedding ===")
    sentence = "The queen rules the country."
    doc_vec = embedder.embed_document(sentence)
    print(f"Sentence: {sentence}")
    print(f"Vector shape: {doc_vec.shape}")
    print(doc_vec)


if __name__ == "__main__":
    main()
