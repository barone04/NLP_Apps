import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors
from typing import List, Tuple, Optional
from numpy import ndarray
from src.preprocessing.regex_tokenizer import RegexTokenizer


class WordEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model: KeyedVectors = api.load(model_name)
        self.vector_size = self.model.vector_size
        self.tokenizer = RegexTokenizer()

    def get_vector(self, word: str) -> Optional[ndarray]:
        if word in self.model.key_to_index:
            return self.model[word]
        else:
            print(f"Word '{word}' not found in the vocabulary (OOV).")
            return None

    def get_similarity(self, word1: str, word2: str) -> Optional[float]:
        if word1 not in self.model.key_to_index:
            print(f"Word '{word1}' not found in the vocabulary.")
            return None
        if word2 not in self.model.key_to_index:
            print(f"Word '{word2}' not found in the vocabulary.")
            return None

        return float(self.model.similarity(word1, word2))

    def get_most_similar(self, word: str, top_n: int = 10) -> Optional[List[Tuple[str, float]]]:
        if word not in self.model.key_to_index:
            print(f"Word '{word}' not found in the vocabulary.")
            return None

        similar_words = self.model.most_similar(word, topn=top_n)
        return similar_words

    def embed_document(self, document: str) -> np.ndarray:
        """"
        Args:
            document (str): The input text document.

        Returns:
            np.ndarray: The averaged embedding vector (size = model.vector_size).
        """
        tokens = self.tokenizer.tokenize(document)
        vectors = []

        for token in tokens:
            vec = self.get_vector(token)
            if vec is not None:
                vectors.append(vec)

        if not vectors:
            # No known words → return zero vector
            return np.zeros(self.vector_size)

        # Compute element-wise mean
        return np.mean(vectors, axis=0)
