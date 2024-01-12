import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class AdvancedSearchEngine:
    def __init__(self, documents):
        self.documents = documents
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)

        # Word2Vec Model
        self.word2vec = self.train_word2vec(documents)

        # LSA Model
        self.lsa_model = TruncatedSVD(n_components=100)
        self.lsa_matrix = self.lsa_model.fit_transform(self.tfidf_matrix)

    def train_word2vec(self, documents):
        tokenized_documents = [word_tokenize(doc.lower()) for doc in documents]
        model = Word2Vec(tokenized_documents, vector_size=100, window=5, min_count=1, workers=4)
        return model

    def search(self, query, top_n=5):
        # Process query for TF-IDF and Word2Vec
        query_tfidf = self.tfidf_vectorizer.transform([query])
        query_tokens = word_tokenize(query.lower())
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
