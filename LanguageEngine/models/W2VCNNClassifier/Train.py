from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from nltk import word_tokenize, sent_tokenize
import numpy as np
import json
import tensorflow as tf
import tflearn
import random
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.conv import conv_1d, max_pool_1d, global_max_pool


WORD_VEC_SIZE = 100

def prepareWord2VecData(facultyDataLoc="../../../KnowledgeEngine/Data/FacultyDetails.json", contextDataLoc="../TrainingData/context.json"):
    with open(facultyDataLoc, encoding="utf8") as jsonData:
        facultyData = json.load(jsonData)

    with open(contextDataLoc) as jsonData:
        contextData = json.load(jsonData)

    trainingDataArray = []

    for i in range(1, len(facultyData) + 1):
        # trainingDataArray.append(simple_preprocess(facultyData[str(i)]["description"]))
        description = facultyData[str(i)]["description"]
        interests = "Their interests include " ", ".join(facultyData[str(i)]["Interest"]) + "."
        facultyString = description + ". " + interests + "."
        trainingDataArray.append(simple_preprocess(facultyString))

    for intent in contextData["contexts"]:
        sentences = ". ".join([sentence for sentence in intent["patterns"]]) + "."
        trainingDataArray.append(simple_preprocess(sentences))

    return trainingDataArray


def trainWord2Vec(saveLoc="data/"):
    trainingData = prepareWord2VecData()
    model = Word2Vec(trainingData, size=WORD_VEC_SIZE, window=10, min_count=3, workers=10)
    model.train(trainingData, total_examples=len(trainingData), epochs=10)
    model.save(saveLoc + "word2vec.model")


def loadWord2VecModel(modelLoc="data/word2vec.model"):
    return Word2Vec.load(modelLoc)


def prepareCNNData(contextDataLoc="../TrainingData/context.json"):
    # adding zeroes to the end to pad sentences
    
    with open(contextDataLoc, encoding="utf8") as jsonData:
        contextData = json.load(jsonData)

    ## Taking and educated guess by looking at training data that the max length of input won't exceed 30 words

    maxLength = 30
    trainingData = []
    classes = []
    w2vModel = loadWord2VecModel()
    
    for context in contextData["contexts"]:
        classes.append(context["tag"])
    
    for context in contextData["contexts"]:
        classRow = np.zeros((len(classes)))
        classRow[classes.index(context["tag"])] = 1
        for pattern in context["patterns"]:
            temp = []
            for word in simple_preprocess(pattern):
                try:
                    temp.append(np.array(w2vModel.wv[word]))
                except Exception:
                    temp.append(np.zeros((WORD_VEC_SIZE)))
            
            # padding the sentences
            temp = padSentenceVector(temp, maxLength, size=WORD_VEC_SIZE)
            trainingData.append([temp, classRow])

    random.shuffle(trainingData)
    trainingData = np.array(trainingData)
    trainingX = list(trainingData[:, 0])
    trainingY = list(trainingData[:, 1])

    return trainingX, trainingY, maxLength
    # cnnModel = makeCNNModel(vectorSize=WORD_VEC_SIZE, sentLength=maxLength, len(classes))


def makeCNNModel(vectorSize, sentLength, numClasses):
    
    # tf.reset_default_graph()
    # net = input_data(shape=[None, sentLength])
    # # net = tf.expand_dims(net, 2)
    # net = conv_1d(net, 128, 5, padding="valid", activation='relu')
    # net = max_pool_1d(net, 2)
    # net = dropout(net, 0.5)
    # net = fully_connected(net, numClasses, activation='softmax')
    # net = regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')
    network = input_data(shape=[None, sentLength, vectorSize], name='input')
    network = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
    # branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
    # branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
    # network = tflearn.merge([branch1, branch2, branch3], mode='concat', axis=1)
    # network = tf.expand_dims(network, 2)
    network = max_pool_1d(network, 2)
    network = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
    network = max_pool_1d(network, 2)
    network = dropout(network, 0.5)
    network = fully_connected(network, numClasses, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.0001,
    loss='categorical_crossentropy', name='target')
    model = tflearn.DNN(network, tensorboard_dir='data/tflearn_logs')
    return model


def trainCNN():
    trainingX, trainingY, sentLength = prepareCNNData()
    print(len(trainingY[0]))
    print(len(trainingX))
    model = makeCNNModel(WORD_VEC_SIZE, sentLength, len(trainingY[0]))
    model.fit(trainingX, trainingY, n_epoch=100000, batch_size=16, show_metric=True)
    model.save('data/TrainedModel/model.tflearn')

    

def padSentenceVector(vector, length, size=WORD_VEC_SIZE):
    if len(vector) is length:
        return vector
    else:
        for i in range(length - len(vector)):
            vector.append(np.zeros(size))
        return vector

trainCNN()

# with open("../TrainingData/context.json") as jsonData:
#     data = json.load(jsonData)

# sentences = 0
# for context in data["contexts"]:
#     sentences += len(context["patterns"])
# print(sentences)