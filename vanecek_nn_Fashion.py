import numpy as np
import keras
from keras import backend as K
import random
from matplotlib import pyplot as plt
K.set_image_dim_ordering('tf')

#data is list of list of datas

def num_to_pic(pixs):
    p=''
    for row in pixs:
        for pix in row:
            if   pix < 0.01:
                p+=' '
            elif pix < 0.21:
                p+='\''
            elif pix < 0.41:
                p+='-'
            elif pix < 0.61:
                p+='o'
            elif pix <0.81:
                p+='#'
            else:
                p+='@'
        p+='\n'
    return p

def plot(data, name, title):
    plt.clf()
    plt.ion()
    with open('final_output_MNIST.txt','a') as f:
        f.write(str(title)+'\n\n\n')
        for i in range(len(data)-3):
            plt.plot(data[i], label=name[i])
        for i in range(len(data)):
            f.write(str(name[i])+'\n'+str(data[i])+'\n\n\n')

    plt.legend()
    plt.title(title)
    plt.savefig(title+'.png', bbox_inches='tight')


x_train = np.load('trainImages.npy')
y_train = np.load('trainLabels.npy')
x_test  = np.load('testImages.npy')
y_test  = np.load('testLabels.npy')
num_classes = np.shape(y_train)[1]
input_shape = np.shape(x_train[0])
x_train = x_train/255
x_test  = x_test/255 #scale the data to [0,1]

'''
x_train = x_train[:500]
y_train = y_train[:500]
x_test=x_test[:500]
y_test=y_test[:500]
'''
batchnorm               =  False
total_conv_layers_dim   =[  [[3,3],[3,3], [5,5], [5,5]] ]#[[5,5],[5,5]] ,         [[3,3],[3,3],[3,3]]      ]
total_conv_output_dim   =[ [32,32,64,64] ]#[32,64] ,                  [128,64,64]     ]
total_FC_dim            =[ [64,16,10] ] #,           [1000]          ]
total_dropout_layers    =[ [0.4,0.4, 0.4,0.4] ]#,     [0.0,0.0]               ]
batch_size              =   64
epochs                  =   50

for conv_layers_dim_i in range(len(total_conv_layers_dim)):
    conv_layers_dim = total_conv_layers_dim[conv_layers_dim_i]
    conv_output_dim = total_conv_output_dim[conv_layers_dim_i]
    for FC_dim in total_FC_dim:
        #last_name_loss = []
        #last_name_name = []
        for dropout_layers in total_dropout_layers:
            #last_name="dropout:"+str(dropout_layers)#+",batchsize:"+str(batch_size)
            #last_name=last_name.replace(' ','')
            #last_name_name.append(last_name)
            first_name = "NOTHINGconvDim"+str(conv_layers_dim)+ ",outpDim"+str(conv_output_dim)+ ",FCDim"+str(FC_dim)+",dropout"+str(dropout_layers)
            first_name = first_name.replace(' ', '')
            print(first_name)
            model = keras.Sequential()
            model.add(keras.layers.Conv2D(conv_output_dim[0], kernel_size=(conv_layers_dim[0][0], conv_layers_dim[0][1]), activation='relu', input_shape=input_shape, data_format='channels_first'))
            #model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), data_format='channels_first'))#, strides=(2, 2)))
            if batchnorm: model.add(keras.layers.BatchNormalization())
            for layer_num in range(1, len(conv_layers_dim)):
                model.add(keras.layers.Conv2D(conv_output_dim[layer_num], (conv_layers_dim[layer_num][0], conv_layers_dim[layer_num][1]), activation='relu', data_format='channels_first'))
                if batchnorm: model.add(keras.layers.BatchNormalization())
                if layer_num%2==0:
                    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), data_format='channels_first'))
                    model.add(keras.layers.Dropout(0.25))
                

            model.add(keras.layers.Flatten())
            for layer_num in range(len(FC_dim)):
                model.add(keras.layers.Dense(FC_dim[layer_num], activation='relu'))
                if batchnorm: model.add(keras.layers.BatchNormalization())
                model.add(keras.layers.Dropout(dropout_layers[layer_num]))

            model.add(keras.layers.Dense(num_classes, activation='softmax'))
            print(model.summary())
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.SGD(lr=0.01),#, decay=1e-6,momentum=0.9, nesterov=True),
                          metrics=['accuracy'])

            #history=LossHistory()
            #model.history #dict with ['loss'] ['acc']
                            #['val_loss'] ['val_acc']
            history=model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      validation_data=(x_test, y_test))#,
                      #callbacks=[history])

            score = model.evaluate(x_test, y_test, verbose=1)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
            #print('losses:', model.history.losses)
            #last_name_loss.append(history.losses)
            hist = model.history.history
            toprintscores=[  [min(1, x) for x in hist['loss']]  , hist['acc'],hist['val_loss'],hist['val_acc']]
            toprintnames=['train loss', 'train accuracy', 'test loss', 'test accuracy']
            p = model.predict(x_test)
            incorrects=''
            num_total=len(p)
            num_wrong = 0
            for i in range(len(p)):
                y_index = np.argmax(y_test[i])
                p_index = np.argmax(p[i])
                if y_index != p_index:
                    num_wrong +=1
                    if p[i][p_index]>0.9999: #its confident enough
                        incorrects+='Index '+str(i)+', classified as '+str(p_index)+', real is '+str(y_index)+' with confidence of '+str(p[i][p_index])+':\n'+num_to_pic(x_test[i][0])+'\n\n'
                    elif p[i][p_index]<0.33: #its confident no idea
                        incorrects+='Index '+str(i)+', classified as '+str(p_index)+', real is '+str(y_index)+' with confidence of '+str(p[i][p_index])+':\n'+num_to_pic(x_test[i][0])+'\n\n'
                else:
                    if random.random()<0.0005:
                        print('Index '+str(i)+', classified as '+str(p_index)+', real is '+str(y_index)+' with confidence of '+str(p[i][p_index])+':\n'+num_to_pic(x_test[i][0])+'\n\n')
            toprintscores.append(model.summary())
            toprintnames.append('MODEL SUMMARY')
            toprintscores.append(incorrects)
            toprintnames.append('INCORRECTLY CLASSIFIED')
            toprintscores.append(model.get_weights())
            toprintnames.append('MODEL WEIGHTS')
            plot(toprintscores, toprintnames, first_name)
