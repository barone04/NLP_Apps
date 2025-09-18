from abc import ABC, abstractmethod


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        pass

class Vectorizer(ABC):
    @abstractmethod
    def fit(self, corpus: list[str]):
        pass

    @abstractmethod
    def transform(self, documents: list[str]) -> list[list[int]]:
        pass

    @abstractmethod
    def fit_transform(self, corpus: list[str]) -> list[list[int]]:
        pass


