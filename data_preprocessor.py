import re
import nltk
nltk.download('wordnet')


def preprocess_data(comments):
    stemmer = nltk.stem.WordNetLemmatizer()
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
        # Removing prefixed '&gt;', '&lt;', '&amp', 'tl;dr', '/r/', '/u/', 'http', 'https', 'www', 'com',
        #                   'youtube.com/watch', 'youtu.be'
        #                   'Very short discussion posts are usually a sign...'
        comment = re.sub(
            r'&gt;|&lt;|&amp;|tl;dr|https|http|www|\.com|np\.reddit|reddit|youtube\.com/watch|youtu\.be|/\S/|very short discussion((.|\n)*)the bot\.|this is ((.|\n)*)original]',
            ' ', comment)
        # Remove all the special characters
        comment = re.sub(r'\W', ' ', comment)
        # Lemmatization
        comment = comment.split()
        comment = [stemmer.lemmatize(word, 'v') for word in comment]
        comment = ' '.join(comment)
        comment = re.sub(r'', '', comment)
        # remove all single characters
        comment = re.sub(r'\s+[a-zA-Z]\s+', ' ', comment)
        # Remove single characters from the start
        comment = re.sub(r'\^[a-zA-Z]\s+', ' ', comment)
        # Substituting multiple spaces with single space
        comment = re.sub(r'\s+', ' ', comment, flags=re.I)
        preprocessed_comments.append(comment)

    return preprocessed_comments
