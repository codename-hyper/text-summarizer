import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

# getting stopwords and punctuations to remove from doc
stopwords = list(STOP_WORDS)
punctuation = punctuation + '\n'

# nlp model
nlp = spacy.load('en_core_web_sm')

# example text
text = 'The adjective "deep" in deep learning comes from the use of multiple layers in the network. Early work showed that a linear perceptron cannot be a universal classifier, and then that a network with a nonpolynomial activation function with one hidden layer of unbounded width can on the other hand so be. Deep learning is a modern variation which is concerned with an unbounded number of layers of bounded size, which permits practical application and optimized implementation, while retaining theoretical universality under mild conditions. In deep learning the layers are also permitted to be heterogeneous and to deviate widely from biologically informed connectionist models, for the sake of efficiency, trainability and understandability, whence the "structured" part. '

# tokenization
doc = nlp(text)

# getting the frequency of each word in doc
count_words = {}
for token in doc:
    if token.text.lower() not in stopwords:
        if token.text.lower() not in punctuation:
            if token.text.lower() not in count_words.keys():
                count_words[token.text.lower()] = 1
            else:
                try:
                    count_words[token.text.lower()] += 1
                except:
                    print('error')

max_count = max(count_words.values())

normalized_count = count_words
for key in normalized_count.keys():
    normalized_count[key] = normalized_count[key] / max_count

sentences = [sent for sent in doc.sents]
print(sentences)
