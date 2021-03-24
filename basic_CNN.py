from utils.models import CNN_Model
from utils.data_loading import prepare_features
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data_path = 'data/'
classes = 4
channels = 22
crossValidation = False
batch_size=32
lr_plateau = ReduceLROnPlateau(patience=4)
early_stopper = EarlyStopping(patience=20)
optimizer = Adam(lr=0.001)

for subject in range(1):
    path = data_path+'s{:}/'.format(subject+1)
    X_train,y_train,y_train_onehot,X_test,y_test,y_test_onehot = prepare_features(path,subject,crossValidation)
    model=CNN_Model(nb_classes=classes,Chans=channels,Samples=1125)
    model.summary()
    model.compile(loss=categorical_crossentropy,optimizer=optimizer,metrics=['accuracy'])

    #train model
    history=model.fit(x=X_train,y=y_train_onehot,batch_size=batch_size,epochs=20)

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


    #测试集准确率
    y_pred = model.predict(X_test).argmax(axis=-1)
    print(y_pred)
    labels = y_test_onehot.argmax(axis=-1)
    accuracy_of_test = accuracy_score(labels, y_pred)
    print(accuracy_of_test)
