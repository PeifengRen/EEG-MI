
from keras import  models
from Idea2.Processing import prepare_train_test_data
from  keras.layers import Input,Dense,Activation,BatchNormalization,Flatten
from keras.layers import SeparableConv2D,Conv2D,Convolution2D,concatenate,AveragePooling2D,Dropout,LSTM
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
import numpy as np
from sklearn.metrics import accuracy_score


def create_model():
    #位置矩阵数据模型
    input1=Input((6,7,250))
    block1=SeparableConv2D(filters=32,kernel_size=(3,3),strides=(1,1),activation='elu',padding='same',data_format='channels_last')(input1)
    block2=Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='elu',padding='same',data_format='channels_last')(block1)
    block3=Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),activation='elu',padding='SAME')(block2)
    BN1=BatchNormalization(axis=-1)(block3)
    block4=Conv2D(13,(1,1),strides=(1,1),activation='elu',padding='same')(BN1)
    cnn1_output=Flatten()(block4)

    #导联相关性矩阵模型输入
    input2=Input((22,22,1))

    cnn2_block1=Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),activation='elu',padding='same',data_format='channels_last')(input2)
    cnn2_BN1=BatchNormalization(axis=-1)(cnn2_block1)
    cnn2_block2=Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='elu',padding='same')(cnn2_BN1)
    cnn2_BN2 = BatchNormalization(axis=-1)(cnn2_block2)
    cnn2_block3=AveragePooling2D((2,2),data_format='channels_last')(cnn2_BN2)
    cnn2_block4 = Dropout(0.1)(cnn2_block3)
    cnn2_output=Flatten()(cnn2_block4)

    x=concatenate([cnn1_output,cnn2_output],axis=-1)

    dense=Dense(4,activation='softmax')(x)



    model=Model(inputs=[input1,input2],outputs=[dense])
    model.summary()
    return model
def cnn_lstm_model():
    cnn_input = Input(shape=(6, 7, 128), name='cnn_input')  # cnn_input
    cnn_1 = SeparableConv2D(32, (4, 4), strides=(1, 1),
                            activation='elu'
                            , padding='SAME'
                            )(cnn_input)
    cnn_2 = Convolution2D(64, (4, 4), strides=(1, 1),
                          activation='elu'
                          , padding='SAME'
                          )(cnn_1)
    cnn_3 = Convolution2D(128, (4, 4), strides=(1, 1),
                          activation='elu'
                          , padding='SAME'
                          )(cnn_2)
    cnn_4 = Convolution2D(13, (1, 1), strides=(1, 1),
                          activation='elu', padding='SAME'
                          )(cnn_3)
    cnn_output = Flatten()(cnn_4)
    lstm_input = Input(shape=(22, 128), name='lstm_input')  # lstm_input
    dense = Dense(1024, input_shape=(22, 128))(lstm_input)
    lstm_1 = LSTM(32, input_shape=(32, 1024), return_sequences=True)(dense)
    lstm_2 = LSTM(32)(lstm_1)
    x = concatenate([cnn_output, lstm_2], axis=-1)
    # output_2=Flatten()(output_1)
    output = Dense(4, activation='softmax')(x)  # output
    model = Model(inputs=[cnn_input, lstm_input], outputs=[output])
    # print(type(model))
    model.summary()
    return model

if __name__ == '__main__':
    data_path = 'D:\PyProjects\eeg-tcnet\data/'
    for subject in range(2, 3):
        path=data_path + 's{:}/'.format(subject + 1)
        position_array_train,position_label_train,lstm_array_train,lstm_label_train,mutual_array_train,mutual_label_train,\
        position_array_test,position_label_test,lstm_label_test,lstm_label_test,mutual_array_test,mutual_label_test=prepare_train_test_data(path,subject,crossValidation=False,window_size=128,overlap=64)

        y_train=(position_label_train-1).astype(int)
        y_test=(position_label_test-1).astype(int)

        #随机划分数据(shuffle_data)
        #Train:
        permutation_train=np.random.permutation(y_train.shape[0])
        y_train=y_train[permutation_train]
        position_array_train=position_array_train[permutation_train]
        lstm_array_train=lstm_array_train[permutation_train]
        mutual_array_train=mutual_array_train[permutation_train]


        #Test:
        permutation_test=np.random.permutation(y_test.shape[0])
        y_test=y_test[permutation_test]
        position_array_test=position_array_test[permutation_test]
        lstm_array_test=lstm_array_test[permutation_test]
        mutual_array_test=mutual_array_test[permutation_test]

        #one-hot
        y_train_onehot= to_categorical(y_train)
        y_test_onehot=to_categorical(y_test)

        # #CNN：position——mutual Model
        # model=create_model()
        # opt = Adam(lr=0.01)
        # model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
        # model.fit([position_array_train,mutual_array_train],y_train_onehot,batch_size=64,epochs=750,verbose=1)
        #
        # y_pred=model.predict([position_array_test,mutual_array_test]).argmax(axis=-1)
        # print(y_pred)
        # labels = y_test_onehot.argmax(axis=-1)
        # accuracy_of_test = accuracy_score(labels, y_pred)
        #
        # print(accuracy_of_test)

        cnn_lstm_Model=cnn_lstm_model()
        opt = Adam(lr=0.01)
        cnn_lstm_Model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
        history=cnn_lstm_Model.fit([position_array_train,lstm_array_train],y_train_onehot,batch_size=64,epochs=750,verbose=1)
        y_pred=cnn_lstm_Model.predict([position_array_test,lstm_array_train]).argmax(axis=-1)
        print(y_pred)
        labels = y_test_onehot.argmax(axis=-1)
        accuracy_of_test = accuracy_score(labels, y_pred)

        print(accuracy_of_test)





