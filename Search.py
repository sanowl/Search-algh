import numpy as np
import nltk
import gensim
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec, Phrases
from sklearn.decomposition import TruncatedSVD

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class AdvancedSearchEngine:
    def __init__(self, documents):
        self.documents = documents
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = RegexpTokenizer(r'\w+')

        # Enhanced Text Normalization
        normalized_documents = [self.normalize_text(doc) for doc in documents]

        # TF-IDF Model
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(normalized_documents)

        # Phrase Detection and Word2Vec Model
        self.word2vec = self.train_word2vec(normalized_documents)

        # LSA Model
        self.lsa_model = TruncatedSVD(n_components=100)
        self.lsa_matrix = self.lsa_model.fit_transform(self.tfidf_matrix)

    def normalize_text(self, text):
        # Lowercasing and Tokenizing
        tokens = self.tokenizer.tokenize(text.lower())
        return ' '.join([token for token in tokens if token not in self.stop_words])

    def train_word2vec(self, documents):
        tokenized_documents = [self.tokenizer.tokenize(doc.lower()) for doc in documents]
        phrases = Phrases(tokenized_documents, min_count=1, threshold=10)
        bigram = gensim.models.phrases.Phraser(phrases)
        bigram_documents = [bigram[doc] for doc in tokenized_documents]
        model = Word2Vec(bigram_documents, vector_size=100, window=5, min_count=1, workers=4)
        return model

    def search(self, query, top_n=5):
        # Normalizing and Processing Query
        normalized_query = self.normalize_text(query)
        query_tfidf = self.tfidf_vectorizer.transform([normalized_query])
        query_tokens = self.tokenizer.tokenize(normalized_query.lower())
        query_vec = np.mean([self.word2vec.wv[token] for token in query_tokens if token in self.word2vec.wv], axis=0)

        # Calculate cosine similarity for TF-IDF and LSA
        cosine_similarities_tfidf = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        query_lsa = self.lsa_model.transform(query_tfidf)
        cosine_similarities_lsa = cosine_similarity(query_lsa, self.lsa_matrix).flatten()

        # Combine and rank results
        combined_scores = (cosine_similarities_tfidf + cosine_similarities_lsa) / 2
        top_indices = np.argsort(combined_scores, axis=0)[-top_n:][::-1]

        top_documents = [(self.documents[i], combined_scores[i]) for i in top_indices]
        return top_documents

# Sample usage
documents = ["your document text here", "another document text", "..."]  # Replace with your dataset
search_engine = AdvancedSearchEngine(documents)
search_results = search_engine.search("your query here")  # Replace with your query

for doc, score in search_results:
    print(f"Document: {doc}\nScore: {score}\n")
