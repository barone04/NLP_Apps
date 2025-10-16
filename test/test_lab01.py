from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.core.dataset_loaders import load_raw_text_data

simple_tokenizer = SimpleTokenizer()
regex_tokenizer = RegexTokenizer()

test_sentences = [
    "Hello, world! This is a test.",
    "NLP is fascinating... isn't it?",
    "Let's see how it handles 123 numbers and punctuation!"
]

print("================= Task 1,2 ====================")
for sentence in test_sentences:
    print(f"Sentence: {sentence}")
    print(f"SimpleTokenizer: {simple_tokenizer.tokenize(sentence)}")
    print(f"RegexTokenizer: {regex_tokenizer.tokenize(sentence)}\n")


dataset_path = "https://github.com/UniversalDependencies/UD_English-EWT.git"
raw_text = load_raw_text_data(dataset_path)

sample_text = raw_text[:150]

simple_tokenizer = SimpleTokenizer()
regex_tokenizer = RegexTokenizer()

print("=================== Task 3 =====================")
print(f"Original Sample: {sample_text[:100]}...")
simple_tokens = simple_tokenizer.tokenize(sample_text)
print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")
regex_tokens = regex_tokenizer.tokenize(sample_text)
print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")