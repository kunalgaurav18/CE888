from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Config import *

'''This is a helper file which helps in loading the images efficiently.'''
def img_loader():
    '''Create the image loaders used to load images from a folder.'''
    # ImageDataGenerator with all the image augmentations to be applied on the training set.
    train_image_gen = ImageDataGenerator(rotation_range=180, width_shift_range=0.1,
                                         height_shift_range=0.1,
                                         horizontal_flip=True,
                                         vertical_flip=True,
                                         rescale=1.0 / 255,
                                         zoom_range=0.1,
                                         fill_mode='nearest',
                                         validation_split=0.2)
    # ImageDataGenerator without any augmentation technique for test set
    test_image_gen = ImageDataGenerator(rescale=1.0 / 255)

    return train_image_gen, test_image_gen


def img_generator():
    '''This method creates and returns the ImageDataGenerator objects which is used for training, validating and
    evaluating the model'''
    train_image_gen, test_image_gen = img_loader()
    train_generator = train_image_gen.flow_from_directory(train_path,
                                                          target_size=img_shape[:2],
                                                          color_mode='rgb',
                                                          class_mode='binary',
                                                          batch_size=batch,
                                                          shuffle=True,
                                                          seed=42,
                                                          subset='training')

    val_generator = train_image_gen.flow_from_directory(train_path,
                                                        target_size=img_shape[:2],
                                                        color_mode='rgb',
                                                        class_mode='binary',
                                                        batch_size=batch,
                                                        shuffle=True,
                                                        seed=42,
                                                        subset='validation')

    test_generator = test_image_gen.flow_from_directory(test_path,
                                                        target_size=img_shape[:2],
                                                        color_mode='rgb',
                                                        class_mode='binary',
                                                        shuffle=False,
                                                        batch_size=batch)

    return train_generator, val_generator, test_generator
