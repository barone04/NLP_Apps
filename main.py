from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer


simple_tokenizer = SimpleTokenizer()
regex_tokenizer = RegexTokenizer()

test_sentences = [
    "Hello, world! This is a test.",
    "NLP is fascinating... isn't it?",
    "Let's see how it handles 123 numbers and punctuation!"
]

for sentence in test_sentences:
    print(f"Sentence: {sentence}")
    print(f"SimpleTokenizer: {simple_tokenizer.tokenize(sentence)}")
    print(f"RegexTokenizer: {regex_tokenizer.tokenize(sentence)}\n")