from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_data= train_datagen.flow_from_directory(r'/content/drive/MyDrive/train-20221106T023729Z-001/train',
                                target_size=(80,80),batch_size=8,class_mode='categorical',subset='training',color_mode='grayscale')
test_data = test_datagen.flow_from_directory(r'/content/drive/MyDrive/test-20221106T023808Z-001/test',
                                target_size=(80,80),batch_size=8,class_mode='categorical',color_mode='grayscale')
