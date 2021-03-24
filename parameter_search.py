"""
网格搜索超参数调优 https://github.com/anthonylauly/Motor-Imagery-CNN-classification/blob/main/cnn_hyperparameter_search.py
https://github.com/shibuiwilliam/keras_grid/blob/master/sample.py
寻找模型最优超参数
"""
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer,Permute,BatchNormalization
from tensorflow.keras.layers import Conv2D, Dense, Flatten,Activation,Dropout,MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import DepthwiseConv2D,SeparableConv2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from utils.data_loading import prepare_features,load_data_by_task
from sklearn.preprocessing import StandardScaler
def variable_init():

    num_learning_rate=[0.001,5e-04,1e-4,5e-5,5e-6]
    num_F1 = [8,16]  # number of temporal filters
    num_ps2D=[6,8]
    # num_FT=[12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    # num_KT=[3,4]
    num_KernLength=[32,64,128]
    num_EEG_dropoutrate=[0.1,0.2,0.25,0.3,0.5]
    # num_TCN_dropoutrate2=[0.1,0.2,0.25,0.3,0.5]
    param_grid=dict(num_learning_rate=num_learning_rate,num_F1=num_F1,num_ps2D=num_ps2D,
                    num_KernLength=num_KernLength)
    return param_grid
def create_model(num_learning_rate=0.001,num_F1=8,num_ps2D=6,
           num_KernLength=64,num_EEG_dropoutrate=0.5):
    model=Sequential()
    model.add(InputLayer(input_shape=(1,22,1125)))
    model.add(Permute((3,2,1)))
    model.add(Conv2D(num_F1,kernel_size=[num_KernLength,1],padding='same',use_bias=False,data_format='channels_last'))
    model.add(BatchNormalization(axis=-1))

    model.add(DepthwiseConv2D((1, 22), use_bias=False,
                             depth_multiplier=2,data_format='channels_last',
                             depthwise_constraint=max_norm(1.)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('elu'))
    model.add(AveragePooling2D((num_ps2D,1),data_format='channels_last'))
    model.add(Dropout(num_EEG_dropoutrate))

    # SeparableConv2D
    model.add(SeparableConv2D(num_F1*2,(16,1),padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('elu'))
    model.add(AveragePooling2D((8,1),data_format='channels_last'))
    model.add(Dropout(num_EEG_dropoutrate))
    model.add(Flatten(name='flatten'))

    model.add(Dense(4,name='dense',kernel_constraint=max_norm(0.25)))
    model.add(Activation('softmax',name='softmax'))

    optimizer=Adam(lr=num_learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model
def parameter_search(X,y,para_grids,num_epochs=100):
    model=KerasClassifier(build_fn=create_model,epochs=num_epochs,batch_size=64)
    grid=GridSearchCV(estimator=model,param_grid=para_grids,cv=5)
    grid_result=grid.fit(X,y)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    params = grid_result.cv_results_['params']
    for mean, param in zip(means, params):
        print("%f  with: %r" % (mean, param))

    return grid_result

data_path='data/'

for subject in range(2,3):
    path = data_path+'s{:}/'.format(subject+1)
    X_train,y_train,y_train_onehot,X_test,y_test,y_test_onehot = prepare_features(path,subject,crossValidation=False)
    print("X_train.shape-----------:",X_train.shape)

    print("___________________________________________")
    for j in range(22): #导联归一化
        scaler = StandardScaler()
        scaler.fit(X_train[:,0,j,:])
        X_train[:,0,j,:] = scaler.transform(X_train[:,0,j,:])
        X_test[:,0,j,:] = scaler.transform(X_test[:,0,j,:])
    #模型输入样本大小(1,22,1125)
    parameter_search(X_train,y_train_onehot,variable_init(),750)








