# Source: https://www.youtube.com/watch?v=1lwddP0KUEg&list=PL7yh-TELLS1G9mmnBN3ZSY8hYgJ5kBOg-&index=7&ab_channel=NeuralNine
# It is my way to learn Machine Learning ( code + mini-conspect)
import random
import json    # there are the list of strings ( questions and answers)
import pickle  # serialization and deserialization library (Serialization is the process of turning an object in memory into a stream of bytes so you can do stuff like store it on disk or send it over the network)
import numpy as np

import nltk  #NLTK(Natural Language Toolkit) is a leading platform for building Python programs to work with human language data.
from nltk.stem import WordNetLemmatizer # Lemmatization is the process of grouping together different inflected forms of words having the same root or lemma for better NLP analysis and operations.
from tensorflow.keras.models import Sequential # Sequential groups a linear stack of layers into a tf.keras.Model and  provides training and inference features on this model
from tensorflow.keras.layers import Dense, Activation, Dropout # Dence - regular densely-connected NN layer; Activation - Applies an activation function to an output; Dropout - Dropout is one of the regularization methods (combating model overfitting). The point of dropout is to "forget" a piece of information. Those. some predetermined percentage of neural connections is broken (forgotten) at the output of the current layer of the neural network. So instead of trying to fit the weights perfectly to just the training dataset, the neural network learns to fit the answer to similar data that did not occur in the training set.
from tensorflow.keras.optimizers import SGD # Optimization type (you have it in your conspects)

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read()) # read the data from the created before json file

words = []
classes = []
documents = []
ignore_letters = ['?', '!', ',', '.'] # list of letters which would be ignored in time of completion the lists: words, classes, documents

for intent in intents['intents']: # json -> intents
    for pattern in intent['patterns']: # json -> intents -> patterns
        word_list = nltk.word_tokenize(pattern) #  breaks a string of each pettern into a list of words
        words.extend(word_list) # appending elements from the specified iterable
        documents.append((word_list, intent['tag'])) # add to the documents lists of pattern's words in form: (['list', 'of', 'words'], 'tag')
        if intent['tag'] not in classes: # if list 'classes' doesn't contain current pattern's tag, it would be added
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters] # in the list 'words' remove all cognate words, leaving only one
words = sorted(set(words))

classes = sorted(set(classes))

# save lists 'words' and 'classes' in serialized form
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


training = []
output_empty = [0] * len(classes) # if number of tags = 5, then [0 0 0 0 0]

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns] # lemmatize each of the previously added to the documents list word_list
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0) # determine which words from the words list match the current pattern and save the result into the bag list

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row]) # save bag list data with the result of the class type

random.shuffle(training) # shuffle the data to avoid tag placement patterns
training = np.array(training) # training list save as an array of lists

train_x = list(training[:, 0]) # input data
train_y = list(training[:, 1]) # output

model = Sequential() # create the sequental model
# train the model
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=0.000001, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) #optimize the model

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1) #Trains the model for a fixed number of epochs
model.save('chatbotmodel.h5', hist) # save the data in the .h5 form, for the future useful usage in chat bot
print('Done')






