from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
import nltk

pos_to_wornet_dict = {

    'JJ': wn.ADJ,
    'JJR': wn.ADJ,
    'JJS': wn.ADJ,
    'RB': 'a',
    'RBR': 'a',
    'RBS': 'a',
    'NN': 'n',
    'NNP': 'n',
    'NNS': 'n',
    'NNPS': 'n',
    'VB': 'v',
    'VBG': 'v',
    'VBD': 'v',
    'VBN': 'v',
    'VBP': 'v',
    'VBZ': 'v',
    'DT' : 'n',
    'IN' : 'n',
    'PRP$' : 'n',
    'MD' : 'n',
    'TO' : 'n',
    'CC' : 'n',
    'PRP' : 'n'

}


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

cpt = 0
text = ""

with open('subs with time.txt','r') as f:
    for line in f:
        for word in line.split():
            print(word)
            a = nltk.pos_tag(nltk.word_tokenize(word))[0][1]
            tag = 'n'
            if a in pos_to_wornet_dict:
                tag = pos_to_wornet_dict[a]
            newWord = lemmatizer.lemmatize(word, tag)
#            newWord = stemmer.stem(word)
            print(lemmatizer.lemmatize(word, tag))
            if word != newWord:
                cpt = cpt +1
            text = text + newWord
            text = text + " "
print(text)
print(cpt)

file = open("subs stem lem.txt", "w")
file.write(text)
file.close()