from __future__ import print_function
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.models import load_model
import numpy as np
import random
import sys

# load in corpus
path = 'training.txt'
f = open(path).read().lower()

train = f

print('corpus length:', len(train))

chars = sorted(list(set(train)))
print('total words:', len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
maxlen = 40

print('loading model...')
# load model
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")



def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate(sentence):
	#for diversity in [0.2, 0.5, 1.0, 1.2]:

	start_index = random.randint(0, len(train) - maxlen - 1)
	for diversity in [1.0]:
	    print()
	    print('----- diversity:', diversity)

	    generated = ''
	    generated += sentence

	    print('----- Generating with seed: "' + sentence + '"')
	    sys.stdout.write(generated)

	    str_chars = ''
	    for i in range(50):
	        x = np.zeros((1, maxlen, len(chars)))
	        for t, char in enumerate(sentence):
	            x[0, t, char_indices[char]] = 1.

	        preds = model.predict(x, verbose=0)[0]
	        next_index = sample(preds, diversity)
	        next_char = indices_char[next_index]

	        generated += next_char
	        sentence = sentence[1:] + next_char
	    print()
	    print(u""+generated)

while(True):
	feed = raw_input(">")
	feed = feed.decode('utf-8')
	generate(feed)