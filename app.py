import pickle
from flask import Flask, render_template, request
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords


import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


# flas router

app =Flask(__name__)

# load the pickle files
with open('count_vectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)


with open('tfidf_transformer.pkl', 'rb') as f:
    tfidf_transformer = pickle.load(f)

with open('features_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)


# preprocessing data


stop_words = set(stopwords.words('english'))
new_stop_words = ["fig","figure","image","sample","using", 
             "show", "result", "large", 
             "also", "one", "two", "three", 
             "four", "five", "seven","eight","nine"]

stop_words = list(stop_words.union(new_stop_words))


# function for preprocessing data
def preprocess_text(text):
    txt =text.lower()

     # remove html tags
    txt = re.sub(r'<.*?>', " ", txt)
    
    # remove alpha umeric characters
    txt = re.sub(r'[^a-zA-Z]'," ",txt)

   

    # tokenxization
    token_txt = nltk.word_tokenize(txt,  language="english")

   
    # remove stop words
    remove_stop_words = [ word for word in token_txt if word not in stop_words]

    #remove words less than 3 letts
    remove_short_words = [word for word in remove_stop_words if len(word) >= 3]

     # Lemmatization
    lemmatizer = WordNetLemmatizer()
    txt = [lemmatizer.lemmatize(word) for word in remove_short_words]


    return " ".join(txt)




# sparse matrix in coo format 
def sort_coo(coo_matrix):
    """This function sorts the non-zero values of a sparse matrix in COO (Coordinate) format, 
    based on the values in descending order and then by column indices in case of ties.
    """
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key= lambda x: (x[1], x[0]), reverse = True)


def extract_topn_feature_from_vector(feature_names, sorted_items, topn = 10):
    """This function extracts the top N features based on the highest scores from a list of sorted items 
    (typically from a TF-IDF vector or any other vector with importance scores).
    """
    # top items from the vector

    sorted_items = sorted_items[:topn]

    # initialise the list
    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        score_vals.append(round(score,3))
        feature_vals.append(feature_names[idx])

    #create tuples of features, score
    results = {}
    for i in range(len(feature_vals)):
        results[feature_vals[i]] = score_vals[i]
    return results




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract_keywords', methods=['POST','GET'])
def extract_keywords():
    doc =request.files['file']
    # If no file is selected
    if doc.filename =='':
        return render_template('index.html', error = 'No Document Found!')
    
    # If a file is selected
    if doc:
        # file_ext = doc.filename.rsplit('.', 1)[-1].lower()

        # if file_ext in ['txt', 'doc', 'docx']:  # For text-based files
        text = doc.read().decode('utf-8', errors='ignore')
        preprocessed_text = preprocess_text(text)
    
        tf_idf_vector = tfidf_transformer.transform(cv.transform([preprocessed_text]))
        sorted_items = sort_coo(tf_idf_vector.tocoo())
        keywords = extract_topn_feature_from_vector(feature_names, sorted_items, 20)
        return render_template('keywords.html', keywords=keywords)
   
        # else:
        #     return render_template('index.html', error='Unsupported file type!')
    return render_template('index.html')
    

@app.route('/search_keywords', methods = ['POST','GET'])
def search_keywords():
    search_query = request.form['search']
    if search_query:
        keywords = []
        for keyword in feature_names:
            if search_query.lower() in keyword.lower():
                keywords.append(keyword)
                if len(keywords) == 20:
                    break
        return render_template('keywordslist.html', keywords= keywords)
    
    return render_template('index.html')

if  __name__ == '__main__':
    app.run(debug=True)
