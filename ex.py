import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.cluster import AgglomerativeClustering
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import tldextract
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity





def remove_outliers(data, params):
    for column, stats in params.items():
        Q1, Q3, IQR, lb, ub = stats['q1'], stats['q3'], stats['iqr'], stats['lb'], stats['ub']
        data[column] = np.where(data[column] < lb, lb, data[column])
        data[column] = np.where(data[column] > ub, ub, data[column])
    return data

# Initialize tldextract for domain extraction
extractor = tldextract.TLDExtract(cache_dir=False)

def analyze_url(url):
    features = {}
    
    # Parse URL components
    extracted = extractor(url)
    protocol = urlparse(url).scheme  # Get the protocol (http or https)
    print(protocol)
    full_domain = protocol + ('.' + extracted.subdomain if extracted.subdomain else '') + '.' + extracted.domain
    
    # 1. Domain Length (excluding protocol)
    features['DomainLength'] = len(('.' + extracted.subdomain if extracted.subdomain else '') + '.' + extracted.domain)
    
    # 2. Is Domain an IP Address?
    features['IsDomainIP'] = 1 if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', extracted.domain) else 0
    
    # 3. URL Similarity Index (Cosine similarity with risky keywords)
    features['URLSimilarityIndex'] = calculate_similarity_index(url)
    
    # 4. Char Continuation Rate (rate of repeated characters in URL)
    features['CharContinuationRate'] = calculate_continuation_rate(url)
    
    # 5. TLD Legitimate Probability (this could be complex, so it's a placeholder)
    features['TLDLegitimateProb'] = 0.9  # Placeholder example value
    
    # 6. URL Character Probability (basic probability check using character frequencies)
    features['URLCharProb'] = calculate_char_probability(url)
    
    # 7. TLD Length (length of the TLD part)
    features['TLDLength'] = len(extracted.suffix)
    
    # 8. Number of Subdomains
    features['NoOfSubDomain'] = len(extracted.subdomain.split('.')) if extracted.subdomain else 0
    
    # 9. Obfuscation: Check if URL has obfuscation patterns (%20, _, -, etc.)
    features['HasObfuscation'] = 1 if has_obfuscation(url) else 0
    features['NoOfObfuscatedChar'] = len(re.findall(r'%|\.|_|-|@|%20', url))
    features['ObfuscationRatio'] = features['NoOfObfuscatedChar'] / len(url) if len(url) > 0 else 0
    
    # 10. Letters in URL (only letters)
    features['NoOfLettersInURL'] = sum(c.isalpha() for c in url)
    features['LetterRatioInURL'] = features['NoOfLettersInURL'] / len(url) if len(url) > 0 else 0
    
    # 11. Digits in URL (only digits)
    features['NoOfDegitsInURL'] = sum(c.isdigit() for c in url)
    features['DegitRatioInURL'] = features['NoOfDegitsInURL'] / len(url) if len(url) > 0 else 0
    
    # 12. Count specific symbols in URL
    features['NoOfEqualsInURL'] = url.count('=')
    features['NoOfQMarkInURL'] = url.count('?')
    features['NoOfAmpersandInURL'] = url.count('&')
    features['NoOfOtherSpecialCharsInURL'] = len(re.findall(r'[^a-zA-Z0-9:/?&=]', url))
    features['SpacialCharRatioInURL'] = features['NoOfOtherSpecialCharsInURL'] / len(url) if len(url) > 0 else 0
    
    # 13. HTTPS check
    features['IsHTTPS'] = 1 if url.startswith('https://') else 0
    
    # Webpage-specific features (scraping)
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        features['LineOfCode'] = len(soup.prettify().splitlines())
        features['LargestLineLength'] = max(len(line) for line in soup.prettify().splitlines())
        
        title_tag = soup.find('title')
        features['HasTitle'] = 1 if title_tag else 0
        features['DomainTitleMatchScore'] = calculate_domain_title_match_score(title_tag.text if title_tag else '', extracted.domain)
        url_path = urlparse(url).path
        features['URLTitleMatchScore'] = calculate_url_title_match_score(title_tag.text if title_tag else '', url_path)
        
        features['HasFavicon'] = 1 if soup.find('link', rel='icon') else 0
        features['Robots'] = 1 if requests.get(url + '/robots.txt').status_code == 200 else 0
        features['IsResponsive'] = 1 if soup.find('meta', attrs={'name': 'viewport'}) else 0
        features['NoOfURLRedirect'] = len(response.history)
        features['NoOfSelfRedirect'] = len(soup.find_all('a', href=lambda x: x and x.startswith(url)))
        features['HasDescription'] = 1 if soup.find('meta', attrs={'name': 'description'}) else 0
        features['NoOfPopup'] = count_popups(soup)
        features['NoOfiFrame'] = len(soup.find_all('iframe'))
        features['HasExternalFormSubmit'] = 1 if soup.find('form', action=lambda x: x and not x.startswith('/')) else 0
        features['HasSocialNet'] = 1 if soup.find('a', href=lambda x: 'facebook.com' in x or 'twitter.com' in x) else 0
        features['HasSubmitButton'] = 1 if soup.find('input', type='submit') else 0
        features['HasHiddenFields'] = 1 if soup.find('input', type='hidden') else 0
        features['HasPasswordField'] = 1 if soup.find('input', type='password') else 0
        features['Bank'] = 1 if 'bank' in url.lower() else 0
        features['Pay'] = 1 if 'pay' in url.lower() else 0
        features['Crypto'] = 1 if 'crypto' in url.lower() else 0
        features['HasCopyrightInfo'] = 1 if soup.find(text=re.compile(r'copyright', re.I)) else 0
        features['NoOfImage'] = len(soup.find_all('img'))
        features['NoOfCSS'] = len(soup.find_all('link', rel='stylesheet'))
        features['NoOfJS'] = len(soup.find_all('script'))
        features['NoOfSelfRef'] = len(soup.find_all('a', href=lambda x: x and x.startswith(url)))
        features['NoOfEmptyRef'] = len(soup.find_all('a', href=''))
        features['NoOfExternalRef'] = len(soup.find_all('a', href=lambda x: x and not x.startswith(url) and not x.startswith('/')))
        
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    
    return pd.DataFrame([features])

# Helper functions
def calculate_continuation_rate(url):
    continuation_count = sum(url[i] == url[i + 1] for i in range(len(url) - 1))
    return continuation_count / (len(url) - 1) if len(url) > 1 else 0

def calculate_similarity_index(url, keywords=["bank", "login", "secure", "account"]):
    vectorizer = TfidfVectorizer().fit_transform([url] + keywords)
    vectors = vectorizer.toarray()
    similarity_matrix = cosine_similarity(vectors)
    return max(similarity_matrix[0][1:])

def calculate_char_probability(url):
    char_counts = {char: url.count(char) for char in set(url)}
    total_chars = len(url)
    probabilities = {char: count / total_chars for char, count in char_counts.items()}
    return sum(probabilities.values()) / len(probabilities) if probabilities else 0

def has_obfuscation(url):
    return bool(re.search(r'%[0-9A-F]{2}|_|-|@|%20', url))

def calculate_domain_title_match_score(title, domain):
    title_words = set(title.lower().split())
    domain_words = set(domain.lower().split('.'))
    return len(title_words & domain_words) / len(title_words) if title_words else 0

def calculate_url_title_match_score(title, path):
    path_words = set(path.lower().split('/'))
    title_words = set(title.lower().split())
    return len(path_words & title_words) / len(path_words) if path_words else 0

def count_popups(soup):
    # Custom function to count possible popups
    return len(soup.find_all('script', src=lambda x: x and 'popup' in x))

# Loading the encoder 
one_hot_encoder = joblib.load(r'C:/Users/Sejal Hanmante/OneDrive/Documents/GitHub/Phishing-url-detection/Clustering/encoder.joblib')

# Loading the outlier removal joblib file
outlier_removal_params = joblib.load(r"C:/Users/Sejal Hanmante/OneDrive/Documents/GitHub/Phishing-url-detection/Clustering/data_outliers.joblib")

# Loading the Scaler joblib file
scaler = joblib.load(r'C:/Users/Sejal Hanmante/OneDrive/Documents/GitHub/Phishing-url-detection/Clustering/scaler.joblib')

# loading the pca transformer 
pca = joblib.load(r'C:/Users/Sejal Hanmante/OneDrive/Documents/GitHub/Phishing-url-detection/Clustering/pca.joblib')


# Loading agglomerative clustering model 
agg_model_data = joblib.load(r'C:/Users/Sejal Hanmante/OneDrive/Documents/GitHub/Phishing-url-detection/Clustering/agg_clustering_model_with_labels.joblib')


#print(outlier_removal_params)

# Recreate the Agglomerative Clustering model using saved parameters
n_clusters = agg_model_data['model_params']['n_clusters']
linkage = agg_model_data['model_params']['linkage']
#affinity = agg_model_data['model_params']['affinity']
agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)

def predict(url):
    features_df = analyze_url(url)

    