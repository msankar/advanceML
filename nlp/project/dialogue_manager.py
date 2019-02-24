import os
from sklearn.metrics.pairwise import pairwise_distances_argmin

from chatterbot import ChatBot
from utils import *

from sklearn.metrics.pairwise import cosine_similarity

# Notes and reference
# https://djangostars.com/blog/how-to-create-and-deploy-a-telegram-bot/
# https://core.telegram.org/bots#creating-a-new-bot
# https://www.sohamkamani.com/blog/2016/09/21/making-a-telegram-bot/
class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        # HINT: you have already implemented a similar routine in the 3rd assignment.
        #### YOUR CODE HERE ####
        ### REF: nlp/week3/week3-Embeddings.ipynb
        question_vec = question_to_vec(question=question, embeddings=self.word_embeddings, dim=self.embeddings_dim)
        
        #### YOUR CODE HERE ####
        ### REF: See week 3 assignment rank_candidates  nlp/week3/week3-Embeddings.ipynb
        quest_vec = np.array([question_vec])
        candidate_vec = thread_embeddings
        similarityScores = list(cosine_similarity(quest_vec, candidate_vec)[0])
        originalPositionList = [(i, thread_embeddings[i], similarityScores[i]) for i in range(thread_embeddings.shape[0])]
        sortedByCosineDistance = sorted(originalPositionList, key=lambda x:x[2], reverse=True)
        best_thread = sortedByCosineDistance[0][0]   
        return thread_ids[best_thread]


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(paths)
        self.create_chitchat_bot()

    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        # Hint: you might want to create and train chatterbot.ChatBot here.
        # It could be done by creating ChatBot with the *trainer* parameter equals 
        # "chatterbot.trainers.ChatterBotCorpusTrainer"
        # and then calling *train* function with "chatterbot.corpus.english" param
        # https://github.com/gunthercox/ChatterBot
        
        ########################
        #### YOUR CODE HERE ####
        ########################
        #### Snippet from https://github.com/gunthercox/ChatterBot
        #chatbot = ChatBot('MalaSan Bot')
        ## Create a new trainer for the chatbot
        #trainer = ChatterBotCorpusTrainer(chatbot)
        ## Train the chatbot based on the english corpus
        # trainer.train("chatterbot.corpus.english")
        ## Get a response to an input statement
        #chatbot.get_response("Hello, how are you today?"
        
        from chatterbot import ChatBot
        from chatterbot.trainers import ChatterBotCorpusTrainer
        
        self.chitchat_bot = ChatBot(
            'MalaSan Bot',
            trainer='chatterbot.trainers.ChatterBotCorpusTrainer'
        )
        self.chitchat_bot.train("chatterbot.corpus.english")
       
    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.
        # REFERENCE:  week5-project.ipynb
        
        #### YOUR CODE HERE ####
        prepared_question = text_prepare(question) #see utils.py
        #### YOUR CODE HERE ####
        features = self.tfidf_vectorizer.transform([prepared_question])
        #### YOUR CODE HERE ####
        # Pass features
        intent = self.intent_recognizer.predict(features)[0]

        # Chit-chat part:   
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.
            #### YOUR CODE HERE ####
            response = self.chitchat_bot.get_response(question) #https://github.com/gunthercox/ChatterBot
            return response
        
        # Goal-oriented part:
        else:        
            # Pass features to tag_classifier to get predictions.
            #### YOUR CODE HERE ####
            tag = self.tag_classifier.predict(features)[0]
            
            # Pass prepared_question to thread_ranker to get predictions.
            #### YOUR CODE HERE ####
            thread_id = self.thread_ranker.get_best_thread(prepared_question, tag)
           
            return self.ANSWER_TEMPLATE % (tag, thread_id)

