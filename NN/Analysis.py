# -*- coding: utf-8 -*-
import numpy as np
from numpy import loadtxt
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

def Get_X_Y_Train_Test(url):
    
    dataset = loadtxt(url, delimiter=',')
    
    X_cul = dataset.shape[1] - 1
    
    X = dataset[:,0:X_cul]
    y = dataset[:,X_cul]
    
    X, testX, y, testY = train_test_split(X,y,test_size=0.2, random_state=0)
    
    return X, testX, y, testY

def model_structure(optimizer='adam', metrics=['accuracy']):
    
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)
    
    return model


def Get_Best_param(model_structure,X,y,batch_size,epochs):
    
    model = KerasClassifier(build_fn=model_structure, verbose=0)
    # define the grid search parameters
    batch_size  = [10, 20, 40, 60, 80, 100]
    epochs      = [10, 50, 100]
    param_grid  = dict(batch_size=batch_size, epochs=epochs)
    grid        = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X, y)
    
    param_best = grid_result.best_params_
    score_best = grid_result.best_score_
    
    
    batch_size_ = param_best['batch_size']
    epochs_     = param_best['epochs']
    
    return batch_size_,epochs_,score_best


def Create_model(X,y,epochs_,batch_size_,plot_losse_accurancy = True):
    
    model   = model_structure()
    history = model.fit(X, y,validation_split=0.33,epochs=epochs_, batch_size=batch_size_)
    if plot_losse_accurancy:
        pyplot.title  ('Loss / accuracy')
        pyplot.plot   (history.history['loss'], label='train')
        pyplot.plot   (history.history['accuracy'], label='accuracy')
        pyplot.plot   (history.history['val_loss'], label='val_loss')
        pyplot.legend ()
        pyplot.show   ()
    
    return model

def Check_Test(model,testX,testY):
    
    array_pred = np.squeeze(model.predict(testX))
    ans        = np.squeeze(testY)
    
    calc = sum(abs(ans - array_pred) < 0.5)/len(testY)
    
    return calc


# Extract Data
url = r'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'

X, testX, y, testY = Get_X_Y_Train_Test(url)

# get best bacth and epochs for the model

batch_size         = [10, 20, 40, 60, 80, 100]
epochs             = [10, 50, 100]

batch_size_,epochs_,score_best = Get_Best_param(model_structure,X,y,batch_size,epochs)

# build the model

model = Create_model(X,y,epochs_,batch_size_)

#evaluate the keras model

_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

# predict Model
re_shape = np.array([6,135,72,35,0,22.6,0.587,50]).reshape(1,8)

# 1-sum(abs(ans - pred>0.5) )/data_size

check_test = Check_Test(model,testX,testY)
print (check_test)
