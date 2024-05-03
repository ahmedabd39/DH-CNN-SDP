#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow import keras
from numpy import array
from keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM, Conv1D,MaxPooling1D,Dropout
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers import Concatenate

from tensorflow.keras import regularizers

import pandas as pd
import numpy as np
import re


# In[2]:


df = pd.read_csv("xalan_2.4_with_cfgnodsafterover.csv")
df1 = pd.read_csv("jedit_4.1_with_cfgnodstest.csv")


# In[3]:


df1.columns


# In[4]:


df.head()


# In[5]:


import seaborn as sns

sns.countplot(x='bug', data=df)


# In[6]:


x_train = df.drop('bug', axis=1)
x_train = x_train.drop('name', axis=1)
y_train = df['bug']
x_test = df1.drop('bug', axis=1)
x_test = x_test.drop('name', axis=1)
y_test = df1['bug']
X2_train = x_train[[ "wmc", "dit", "noc", "cbo", 'rfc', "lcom", "ca", "ce", "npm","lcom3", "loc", "dam", "moa", "mfa", "cam", "ic", "cbm","amc","max_cc","avg_cc"]].values
X2_test = x_test[[ "wmc", "dit", "noc", "cbo", 'rfc', "lcom", "ca", "ce", "npm","lcom3", "loc", "dam", "moa", "mfa", "cam", "ic", "cbm", "amc","max_cc", "avg_cc"]].values


# In[7]:


from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train)


# In[8]:


X1_train = []
sentences = list(x_train["cfgnodes"])
for sen in sentences:
    X1_train.append(sen)

X3_train = []
sentences = list(x_train["nodes"])
for sen in sentences:
    X3_train.append(sen)


# In[9]:


X1_test = []
sentences = list(x_test["cfgnodes"])
for sen in sentences:
    X1_test.append(sen)

X3_test = []
sentences = list(x_test["nodes"])
for sen in sentences:
    X3_test.append(sen)


# In[10]:


print(X1_train)


# In[11]:


tokenizer = Tokenizer(num_words=4000)
tokenizer.fit_on_texts(X1_train)

tokenizer1 = Tokenizer(num_words=4000)
tokenizer1.fit_on_texts(X1_test)

X1_train = tokenizer.texts_to_sequences(X1_train)
X1_test = tokenizer1.texts_to_sequences(X1_test)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 50

X1_train = pad_sequences(X1_train, padding='post', maxlen=maxlen)
X1_test = pad_sequences(X1_test, padding='post', maxlen=maxlen)


# In[12]:


tokenizer3 = Tokenizer(num_words=4000)
tokenizer3.fit_on_texts(X3_train)

tokenizer4 = Tokenizer(num_words=4000)
tokenizer4.fit_on_texts(X3_test)

X3_train = tokenizer.texts_to_sequences(X3_train)
X3_test = tokenizer1.texts_to_sequences(X3_test)

vocab_size3 = len(tokenizer3.word_index) + 1

maxlen = 50

X3_train = pad_sequences(X3_train, padding='post', maxlen=maxlen)
X3_test = pad_sequences(X3_test, padding='post', maxlen=maxlen)


# In[13]:


print(X3_train.shape)


# In[14]:


from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()

glove_file = open('xalan_2.4_node2vec_embeddings.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions

glove_file.close()

embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


# In[15]:


embeddings_dictionary2 = dict()

glove_file2 = open('xalan_2.4_word2vec_embedding.txt', encoding="utf8")

for line in glove_file2:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary2[word] = vector_dimensions

glove_file2.close()

embedding_matrix2 = zeros((vocab_size3, 100))
for word, index in tokenizer3.word_index.items():
    embedding_vector = embeddings_dictionary2.get(word)
    if embedding_vector is not None:
        embedding_matrix2[index] = embedding_vector


# In[16]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X2_train)

Xtrain_scaled = scaler.transform(X2_train)
Xtest_scaled = scaler.transform(X2_test)


# In[17]:


print(Xtrain_scaled.shape)


# In[18]:


input_1 = Input(shape=(maxlen,))
input_2 = Input(shape=(20,1))
input_3 = Input(shape=(maxlen,))


# In[ ]:





# In[19]:


Xtrain_scaled = np.reshape(Xtrain_scaled,(Xtrain_scaled.shape[0],Xtrain_scaled.shape[1],1))
Xtest_scaled = np.reshape(Xtest_scaled,(Xtest_scaled.shape[0],Xtest_scaled.shape[1],1))


# In[20]:


# node to vector model
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(input_1)
#CNN_Layer_1 = Conv1D(128,5, activation='relu',\
 #                                kernel_regularizer = regularizers.l2(0.0005),\
  #                               bias_regularizer = regularizers.l2(0.0005))(embedding_layer)
CNN_Layer_1 = Conv1D(128,5, activation='relu')(embedding_layer)
CNN_Layer_2 = GlobalMaxPooling1D()(CNN_Layer_1)

CNN_Layer_3 = Dropout(0.8)(CNN_Layer_2)

CNN_Layer_4 = Dense(10, activation='sigmoid')(CNN_Layer_3) #,\
  #                              kernel_regularizer=regularizers.l2(0.001),\
  #                              bias_regularizer=regularizers.l2(0.001),)(CNN_Layer_3)


# In[21]:


# word to vector model
embedding_layer2 = Embedding(vocab_size3, 100, weights=[embedding_matrix2], trainable=False)(input_3)
#CNN_Layer_1 = Conv1D(128,5, activation='relu',\
 #                                kernel_regularizer = regularizers.l2(0.0005),\
  #                               bias_regularizer = regularizers.l2(0.0005))(embedding_layer)
CNN2_Layer_1 = Conv1D(128,5, activation='relu')(embedding_layer2)
CNN2_Layer_2 = GlobalMaxPooling1D()(CNN2_Layer_1)

CNN2_Layer_3 = Dropout(0.8)(CNN2_Layer_2)

CNN2_Layer_4 = Dense(10, activation='sigmoid')(CNN2_Layer_3) #,\
  #                              kernel_regularizer=regularizers.l2(0.001),\
  #                              bias_regularizer=regularizers.l2(0.001),)(CNN_Layer_3)


# In[22]:



model_m1 = Conv1D(filters=128, kernel_size=5)(input_2)
model_m2=MaxPooling1D(pool_size=5 )(model_m1)
model_m3=Flatten()(model_m2)
model_m4=Dropout(0.8)(model_m3)
model_m5=Dense(8, activation='relu')(model_m4)


# In[23]:


#concat_layer = Concatenate()([CNN_Layer_4, model_m5])
concat_layer = Concatenate()([CNN_Layer_4,CNN2_Layer_4, model_m5])
dense_layer_6 = Dense(8, activation='relu')(concat_layer)
dense_layer_7=Dropout(0.8)(dense_layer_6)
output = Dense(2, activation='softmax')(dense_layer_7)
model = Model(inputs=[input_1,input_3, input_2], outputs=output)


# In[24]:


import tensorflow as tf
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())
#tf.keras.utils.plot_model(model, to_file="my_model.png", show_shapes=True)


# In[25]:


history = model.fit(x=[X1_train,X3_train, Xtrain_scaled], y=y_train, batch_size=1024, epochs=100, verbose=1, validation_data=([X1_test,X3_test,Xtest_scaled],y_test))


# In[26]:


#model.save('2inputSample3') 


# In[27]:


import tensorflow as tf
new_model = model# tf.keras.models.load_model('2inputSample3')
new_model.summary()


# In[28]:


score = model.evaluate(x=[X1_test,X3_test, Xtest_scaled], y=y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# In[29]:


import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()


# In[30]:


classes = model.predict(x=[X1_test,X3_test, Xtest_scaled])


# In[31]:


y_classes = classes.argmax(axis=-1)


# In[32]:


y_classes


# In[33]:


true_classes = y_test.argmax(axis=-1)


# In[34]:


true_classes


# In[35]:


from sklearn.metrics import classification_report
print(classification_report(true_classes, y_classes))


# In[36]:


from sklearn.metrics import roc_auc_score
 
# ROC AUC
auc = roc_auc_score(true_classes, y_classes)
print('ROC AUC: %f' % auc)


# In[37]:


from sklearn.metrics import confusion_matrix
#Get the confusion matrix
cf_matrix = confusion_matrix(true_classes, y_classes)
print(cf_matrix)


# In[38]:


group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')


# **other method for cnn **
