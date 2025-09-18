from src.core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        # chèn khoảng trắng quanh dấu câu để tách
        for i in [',', '.', '!', '?']:
            text = text.replace(i, f' {i} ')

        tokens = text.split()
        return tokens