import nltk
import sys
import os
import string
import math
from collections import Counter

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)
    
    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)
    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)
    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)

def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    documents = {}
    for document in os.listdir(directory):
        path = os.path.join(directory, document)
        with open(path) as file:
            content = file.read()
            documents[document] = content
    return documents


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokenized = nltk.tokenize.word_tokenize(document, language='english')
    stop_words = set(nltk.corpus.stopwords.words("english"))
    punctuation = set(string.punctuation)
    for i in range(len(tokenized) - 1, -1, -1):
        tokenized[i] = tokenized[i].lower()
        if tokenized[i] in stop_words or tokenized[i] in punctuation:
            del tokenized[i]

    return tokenized


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words = {}
    n = len(documents)
    for words_list in documents.values():  # count each word
        file_words = set()
        for word in words_list:
            if word in file_words:
                continue
            else:
                words[word] = words.get(word, 0) + 1
                file_words.add(word)

    for word in words:  # apply idf formula
        words[word] = math.log(n / words[word])

    return words


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words) ex. {python: ['alo']}, and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    for each word in the query, count how many times it apperead in the file times idfs
    """
    ranked_files = {}
    for filename, content in files.items():
        word_counter = Counter(content)
        total = 0
        for word in query:
            total += word_counter.get(word, 0) * idfs.get(word, 0)
        ranked_files[filename] = total
    ranks = sorted(list(ranked_files.keys()), key = lambda x:ranked_files[x], reverse = True)
    return ranks[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentences_rank = {}
    for sentence, sentence_list in sentences.items():
        sum_idf = 0
        word_count = 0
        for word in query:
            if word in sentence_list:
                sum_idf += idfs.get(word, 0)
                word_count += 1
        sentences_rank[sentence] = (sum_idf, word_count / len(sentence_list))
    ranks = sorted(list(sentences_rank.keys()), key = lambda x : sentences_rank[x], reverse = True)
    return ranks[:n]



if __name__ == "__main__":
    main()
