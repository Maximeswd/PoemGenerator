# Import necessary libraries
import os
import re
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.utils as ku
from keras.callbacks import ModelCheckpoint, LambdaCallback
from wordcloud import WordCloud
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers


class PoetryGenerator:
    def __init__(self, data_file):
        self.data = open(data_file, encoding="utf8").read()
        self.corpus = self._generate_corpus()
        self.tokenizer = self._fit_tokenizer()
        self.model = None
        self.max_sequence_len = int()

    def _generate_corpus(self):
        corpus = self.data.lower().split("\n")
        return corpus

    def _fit_tokenizer(self):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.corpus)
        return tokenizer
    
    def generate_wordcloud(self, max_font_size=50, max_words=100, 
                           background_color="black", save_fig=False):
        wordcloud = WordCloud(max_font_size=max_font_size, max_words=max_words,
                              background_color=background_color).generate(self.data)
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        if save_fig:
            plt.savefig("WordCloud.png")
        plt.show()

    def _generate_input_sequences(self):
        total_words = len(self.tokenizer.word_index)
        input_sequences = []
        for line in self.corpus:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences,
                                            maxlen=max_sequence_len,
                                            padding='pre'))
        predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
        label = ku.to_categorical(label, num_classes=total_words+1)
        return predictors, label, total_words, max_sequence_len

    def _build_model(self, total_words, max_sequence_len):
        model = Sequential()
        model.add(Embedding(total_words+1, 100, input_length=max_sequence_len-1))
        model.add(Bidirectional(LSTM(150, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(LSTM(100))
        model.add(Dense(total_words+1//2, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dense(total_words+1, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])
        return model

    def train_model(self, epochs=20):
        predictors, label, total_words, max_sequence_len = self._generate_input_sequences()
        model = self._build_model(total_words, max_sequence_len)
        history = model.fit(predictors, label, epochs=epochs, verbose=1)
        self.model = model
        self.max_sequence_len = max_sequence_len
        # self.save_model("poetry_generator_model.lstm6")
        # Save the model and its weights
        # model.save(f"Poems/{seed_text}_pg_model.h5")

    def generate_poetry(self, seed_text, num_words=25):
        # prompts = {
        #     "sad": ["I'm feeling so alone tonight", 
        #             "It feels like the weight of the world is on my shoulders", 
        #             "I wish things could be different"],
        #     "angry": ["I can't believe you did this to me", "I'm so angry I could scream", "Why did you have to go and ruin everything?"],
        #     "happy": ["I'm on top of the world today", "Everything is going my way", "Life is good"],
        #     "nervous": ["My heart is racing and I can't catch my breath", "I feel like something bad is going to happen", "I'm trying to stay calm but it's not working"],
        #     "tired": ["I'm so tired of feeling this way", "I just want to crawl into bed and sleep forever", "Why does everything feel so hard?"],
        #     "hangry": ["I need to eat something or I'll lose my mind", "My stomach is growling so loudly I can't concentrate", "I'm getting hangry, please hurry up and order"],
        #     "confused": ["I don't know what to do or where to go", "I'm feeling lost and overwhelmed", "I wish I had more clarity on this situation"]
        # }
        
        generated_text = list(seed_text)
        
        # Convert the generated text into words based on a space in the list of tokens
        generated_text = "".join(generated_text).split()

        for _ in range(num_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences(
                [token_list], maxlen=self.max_sequence_len-1,
            padding='pre')
            prediction = self.model.predict(token_list, verbose=0)[0]
            index = np.random.choice(len(prediction), p=prediction)
            char = self.tokenizer.index_word[index]
            generated_text.append(char)
        
        # Remove extra spaces
        return " ".join(generated_text)
    
# # Create a new PoetryGenerator object with the path to the data file
# pg = PoetryGenerator('poem.txt')

# # # Generate a word cloud of the text data
# # # pg.generate_wordcloud()
# seed_text = 'I feel sad'

# # Train the model on the input sequences and labels
# pg.train_model(epochs=1)

# # # Generate a poem given a mood
# poem = pg.generate_poetry(seed_text=seed_text, num_words=25)
# # print(poem)
# poem.split('\n')
# # # Save the poem to a text file
# # with open(f'{mood}_poem.txt', 'w') as f:
# #     f.write(poem)


