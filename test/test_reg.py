from sklearn.feature_extraction.text import CountVectorizer
import re

# Define the custom tokenizer
def custom_tokenizer(text):
    # Match pairs of numbers inside brackets as a single token
    pattern = re.compile(r'\[\d+\.\d+, \d+\]')
    matches = pattern.findall(text)
    return matches

# Create the CountVectorizer instance with custom tokenizer
vectorizer = CountVectorizer(tokenizer=custom_tokenizer, ngram_range=(1, 1))

# Example text
text = "[4.0, 3] [4.0, 3] [4.0, 3] [4.0, 3] [4.0, 3] [4.0, 3] [4.0, 3] [4.0, 3] [0.0, 3] [0.0, 3] [0.0, 3] [0.0, 3] [4.0, 3] [4.0, 3] [4.0, 3] [4.0, 3] [0.0, 3] [4.0, 3] [4.0, 3] [4.0, 3] [4.0, 3] [4.0, 3] [4.0, 3]"

# Fit and transform the text
X = vectorizer.fit_transform([text])

# Print the feature names
print(vectorizer.get_feature_names_out())

# Print the transformed matrix
print(X.toarray())
