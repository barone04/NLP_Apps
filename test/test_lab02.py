from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer


tokenizer = RegexTokenizer()
vectorizer = CountVectorizer(tokenizer)

corpus = [
    "I love NLP.",
    "I love programming.",
    "NLP is a subfield of AI."
]

matrix = vectorizer.fit_transform(corpus)
print("Learned Vocabulary:", vectorizer.vocabulary_)
print("Document-Term Matrix:", matrix)