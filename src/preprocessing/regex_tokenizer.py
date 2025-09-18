import re
from src.core.interfaces import Tokenizer

class RegexTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        # Sử dụng regex để khớp các từ (\w+) hoặc các ký tự, không phải là từ không phải khoảng trắng ([^\w\s])
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens