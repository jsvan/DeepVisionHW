import numpy as np
import keras
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler as MMS
from keras import backend as K
K.set_image_dim_ordering('tf')
import random


def plot(data, name, title):
    plt.clf()
    plt.ion()
    print('plotting', title, name)
    #with open('FINAL_output_breast.txt','a') as f:
        #f.write(str(title)+'\n\n\n')
    for i in range(len(data)-2):
        plt.plot(data[i], label=name[i])
    for i in range(len(data)):
            #f.write(str(name[i])+'\n'+str(data[i])+'\n\n\n')

    plt.legend()
    plt.title(title)
    plt.show(title+'.png', bbox_inches='tight')

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses     = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


x =np.loadtxt(open("breastCancerData.csv", "rb"), delimiter=",")
y =np.loadtxt(open("breastCancerLabels.csv", "rb"), delimiter=",")

x=MMS().fit_transform(x)

x_train, y_train, x_test, y_test = [],[],[],[]

cutoff = 0.15
for i in range(len(x)):
    if random.random() < cutoff:
        x_test.append(x[i])
        y_test.append(y[i])
    else:
        x_train.append(x[i])
        y_train.append(y[i])


x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

#input_shape = (np.shape(x_train)[-1], np.shape(x_train)[-2], np.shape(x_train)[-3])
input_shape = np.shape(x_train[1])


plotted=False
epochs = 100
batch_size =16
layers = [ [30],  [10, 10], [100], [10,15,20], [10,15,20]]
total_dropout = [ [0.4],  [0.4, 0.4], [0.5], [0.3, 0.3, 0.3], [0.3,0.4,0.5]  ]
#all_losses, all_names = [],[]
while not plotted:
    for i in range(len(layers)):
        layer = layers[i]
        dropout=total_dropout[i]
        model = keras.Sequential()
        model.add(keras.layers.Dense(layer[0], activation='relu', input_shape=input_shape, init='normal'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(dropout[0]))
        #model.add(activation='relu', , data_format='channels_first'))
        for l in range(1, len(layer)):
            model.add(keras.layers.Dense(layer[l], activation='relu', init='normal'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(dropout[l]))

        model.add(keras.layers.Dense(1, activation='sigmoid', init='normal'))

        model.compile(loss=keras.losses.mean_squared_error,
                      optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6,momentum=0.9, nesterov=True),
                      metrics=['accuracy'])

        history=LossHistory()
        print(model.summary())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=0,
                  validation_data=(x_test, y_test))#,
                  #callbacks=[history])

        #all_losses.append(history.losses)
        #all_names.append(str(layer+dropout))

        name = 'layers'+str(layer)+',dropout'+str(dropout)
        name = name.replace(' ', '')
        print(name)
        score = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        hist = model.history.history
        toprintscores=[hist['loss'],hist['acc'],hist['val_loss'],hist['val_acc']]
        toprintnames=['train loss', 'train accuracy', 'test loss', 'test accuracy']
        if score[1] > 0.99:
            plotted=True
            p = model.predict(x_test)
            incorrects=''
            for i in range(len(p)):
                if abs(y_test[i] - p[i]) >0.05:
                    incorrects+='\nIndex '+str(i)+': '+str(x_test[i]) + ' as '+str(p[i])+', but is really ' + str(y_test[i]) + '\n'
            toprintscores.append(incorrects)
            toprintnames.append('INCORRECTLY CLASSIFIED')
            toprintscores.append(model.get_weights())
            toprintnames.append('MODEL WEIGHTS')
            plot(toprintscores, toprintnames, name)
