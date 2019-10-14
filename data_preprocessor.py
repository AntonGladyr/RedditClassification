import re
import nltk
from contractions import contractions_dict
nltk.download('wordnet')
from nltk.corpus import wordnet


def expand_contractions(text):
    text_ = []
    for word in text.split():
        if word in contractions_dict:
            text_.append(contractions_dict[word])
        else:
            text_.append(word)
    return ' '.join(text_)


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_data(comments):
    # if type(comments) != 'list':
    #     comments = [comments]

    lemmatizer = nltk.stem.WordNetLemmatizer()
    preprocessed_comments = []

    for index in range(0, len(comments)):
        # Converting to Lowercase
        comment = str(comments[index]).lower()

        # removing '**Spoiler Warning:**...'
        comment = re.sub(r'\*\*spoiler warning(.+?)_guide\)\.', ' ', comment)
        # removing 'This is the best tl;dr I could make', 'reduced by... (I'm a bot)', '[**Extended Summary**]'
        comment = re.sub(r'this is(.+?)original]|reduced(.+?)a bot\)|\[\*\*extended(.+?)\]', ' ', comment)
        # removing 'Your submission...'
        comment = re.sub(r'your submission((.|\n)*)list\]\(\/r|\/new\/\)((.|\n)*)', ' ', comment)
        comment = re.sub(r'your submission((.|\n)*)\*i am a((.|\n)*)to=\/r\/|\) if you have(.+?)\.\*', ' ', comment)
        # removing 'i am a bot...'
        comment = re.sub(r'\*i am a((.|\n)*)to=\/r\/|\) if you have(.+?)\.\*', ' ', comment)
        # Removing prefixed '&gt;', '&lt;', '&amp', 'tl;dr', '/r/', '/u/', '_', 'http', 'https', 'www', 'com',
        #                   'youtube.com/watch', 'youtu.be'
        #                   'Very short discussion posts are usually a sign...'
        #                   'all numbers'
        comment = re.sub(
            r'&gt;|&lt;|&amp;|tl;dr|quot|https|http|www|_|\.com|np\.reddit|reddit|youtube\.com/watch|youtu\.be|/\S/|\d+|very short discussion((.|\n)*)the bot\.|this is ((.|\n)*)original]',
            ' ', comment)

        # contraction
        comment = expand_contractions(comment)

        # Remove all the special characters
        comment = re.sub(r'\W', ' ', comment)
        # Lemmatization
        # TODO: fix this part of code for lemmatizing
        comment = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(comment)]
        comment = ' '.join(comment)
        print(index)
        # comment = comment.split()
        # comment = [stemmer.lemmatize(word) for word in comment]
        # comment = [stemmer.lemmatize(word, 'v') for word in comment]
        # comment = ' '.join(comment)
        ############################################################
        # remove all single characters
        comment = re.sub(r'\s+[a-zA-Z]\s+', ' ', comment)
        # remove all two letters words
        comment = re.sub(r'\s+[a-zA-Z][a-zA-Z]\s+', ' ', comment)
        # Remove single characters from the start
        comment = re.sub(r'^[a-zA-Z]\s+', ' ', comment)
        # Remove two letters words from the start
        comment = re.sub(r'^[a-zA-Z][a-zA-Z]\s+', ' ', comment)
        # Remove single characters from the end
        comment = re.sub(r'\s+[a-zA-Z]$', ' ', comment)
        # Remove two letters words from the end
        comment = re.sub(r'\s+[a-zA-Z][a-zA-Z]$', ' ', comment)
        # TODO: fix
        # Remove characters which are not contained in the ASCII character set
        comment = re.sub(r'\s+[^\x00 -\x7F]+\s+', ' ', comment)

        # Substituting multiple spaces with single space
        comment = re.sub(r'\s+', ' ', comment, flags=re.I)
        preprocessed_comments.append(comment)

    return preprocessed_comments
