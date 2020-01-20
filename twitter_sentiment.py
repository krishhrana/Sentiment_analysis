import numpy as np
import os 
import re
from keras.preprocessing import sequence




def token(sentence, remove_vowels=False, remove_repeat=False, minchars=2):
    tokens = []
#   for t in re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\w]+",sentence.lower()):
    for t in re.findall("[a-zA-Z]+",sentence.lower()):

        if len(t)>=minchars:
            if remove_vowels:
                t=removeVovels(t)
            if remove_repeat:
                t=removeRepeat(t)
            tokens.append(t)
    return tokens

VOWELS = ['a', 'e', 'i', 'o', 'u']

def removeRepeat(string):
    return re.sub(r'(.)\1+', r'\1\1', string)     

def removeVovels(string):
    return ''.join([l for l in string.lower() if l not in VOWELS])

if __name__ == '__main__':
    pass

def normalize_matrix(matrix):
    pass



def create_train_data(path, data_col, label_col):
	f=open(path, 'r')
	sentences=f.read().lower()
	sentences=sentences.split('\n')[:-1]

	X_train=[]
	y_train=[]

	for line in sentences:
		line=line.split('\t')
		tokenized_lines = token(line[data_col])

		char_list=[]
		for words in tokenized_lines:
			for ch in words:
				char_list.append(ch)
			char_list.append(' ')
		#print(char_list)
		X_train.append(char_list)

		if line[label_col]=='0':
			y_train.append(0)
		if line[label_column]=='1':
			y_train.append(1)
		if line[label_column]=='2':
			y_train.append(2)


	print(len(y_train))

	y_train=np.asarray(y_train)
	assert(len(X_train) == y_train.shape[0])

	return[X_train, y_train]




def char2num(mapc2n, mapn2c, train_data, max_len):

	char_num=0
	allchars=[]

	for lines in train_data:
		allchars=set(allchars+lines)
		allchars=list(allchars)


	for char in allchars:
		mappingChar2Num[char]=char_num
		mappingNum2Char[char_num]=char
		char_num +=1

	assert(len(allchars)==char_num)

	X_train = []
	for line in train_data:
		char_list=[]
		for letter in line:
			char_list.append(mappingChar2Num[letter])
		#print(no) -- Debugs the number mappings
		X_train.append(char_list)
	print(mappingChar2Num)
	print(mappingNum2Char)
	#Pads the X_train to get a uniform vector
	#TODO: Automate the selection instead of manual input
	X_train = sequence.pad_sequences(X_train[:], maxlen=max_len)

	return [X_train,mappingNum2Char,mappingChar2Num,char_num]


path='/Users/krishrana/Python/Sub-word-LSTM-master/Data/IIITH_Codemixed.txt'
mappingChar2Num={}
mappingNum2Char={}
max_len=280
label_column=3
data_column=1
labels=['0','1','2']
num_classes=3

out=create_train_data(path,data_column, label_column)
X=out[0]
y=out[1]
print('##################### training_data created ########################')
out_1=char2num(mappingChar2Num, mappingNum2Char, X, max_len)
X=out_1[0]
mappingNum2Char=out_1[1]
mappingChar2Num=out_1[2]
max_features=out_1[3]

X=np.array(X)
y=np.array(y).flatten()

print(X.shape)
print(len(y))


from keras.models import Sequential
from keras.preprocessing import sequence
from keras import backend as K
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras import optimizers
from keras.utils import np_utils


model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_len))
model.add(Convolution1D(nb_filter=128, filter_length=3, border_mode='valid',activation='relu',subsample_length=1))
model.add(MaxPooling1D(pool_length=3))

model.add(LSTM(256, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
model.add(LSTM(256, dropout_W=0.2, dropout_U=0.2, return_sequences=False))

model.add(Dense(3))
model.add(Activation('softmax'))


model.summary()

batch_size=32
epoch=50
adam=optimizers.Adam(lr=0.001)
y=np_utils.to_categorical(y, num_classes)

print(y)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, batch_size=batch_size, epochs=epoch, validation_split=0.2)
model.save('sentiNet.pt')









