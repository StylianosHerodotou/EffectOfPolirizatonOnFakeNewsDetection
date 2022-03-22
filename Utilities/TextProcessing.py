import re
import spacy
from string import punctuation
import contractions
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#init items needed for some of the following functions
stemmer = PorterStemmer()
nlp = spacy.load('en_core_web_sm')

stop_words = set(stopwords.words('english'))

# Sad Emoticons
emoticons_sad = {':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<', ':-[', ':-<', '=\\', '=/',
                 '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c', ':c', ':{', '>:\\', ';('}
#HappyEmoticons
emoticons_happy = {':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}', ':^)', ':-D', ':D', '8-D',
                   '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D', '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P',
                   ':-P', ':P', 'X-P', 'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)', '<3'}
#Emoji patterns
emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)

emoticons = emoticons_happy.union(emoticons_sad)

common_word_numbers=(
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen","twenty", "thirty", "forty",
         "fifty", "sixty", "seventy", "eighty", "ninety","hundred", "thousand", "million", "billion", "trillion"
    )


def expand_contractions(text: str) -> str:
    return contractions.fix(text)

def remove_punctuation(text: str, characters_to_not_remove=['^']):
    my_punctuation = punctuation

    for not_remove in characters_to_not_remove:
        my_punctuation = punctuation.replace(not_remove, "")

    return text.translate(str.maketrans("", "", my_punctuation))

# def get_wordnet_pos(word):
#     """Map POS tag to first character lemmatize() accepts"""
#     tag = nltk.pos_tag([word])[0][1][0].upper()
#     tag_dict = {"J": wordnet.ADJ,
#                 "N": wordnet.NOUN,
#                 "V": wordnet.VERB,
#                 "R": wordnet.ADV}

#     return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_text(text):
    # Create a Doc object
    doc = nlp(text)
    lemmatized_words = [token.lemma_ for token in doc]
    return lemmatized_words

def stem_text(words: list):
    stemmed_words = [stemmer.stem(word) for word in words]
    return stemmed_words

#TODO: Add Entiyy recognition
def clean_article(text):
    number_representation = '^'

    # remove new lines
    text = text.replace('\n', ' ').replace('\r', '')
    # to lowecase:
    text = text.lower()
    # expand contractions
    text = expand_contractions(text)
    # replace consecutive non-ASCII characters with a space
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'Ä¶', '', text)
    # remove emojis
    text = emoji_pattern.sub(r'', text)
    # replace numbers with special character, previously removed from text
    text = re.sub(r'^', '', text)
    text = re.sub(r"[+-]?([0-9]*[.,])?[0-9]+", number_representation, text)

    # lemmatize text
    lemmatized_words = lemmatize_text(text)

    ####remove stop words and emoticons:
    clean_words = []
    for word_token in lemmatized_words:
        if (word_token in common_word_numbers):
            clean_words.append(number_representation)
        elif word_token not in stop_words and word_token not in emoticons:
            clean_words.append(word_token)
    # stem words
    clean_words = stem_text(clean_words)

    text = ' '.join(clean_words)

    # removes punctuation
    text = remove_punctuation(text)

    #remove more than one white space.
    text = ' '.join(text.split())
    return text

def clean_articles_in_df(df, article_column_name):
    clean_article_column_name= "clean_"+article_column_name
    clean_articles = list()
    for index, row in df.iterrows():
        dirty = row[article_column_name]
        clean = clean_article(dirty)
        clean_articles.append(clean)
    df[clean_article_column_name] = clean_articles