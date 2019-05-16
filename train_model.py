from cnn import simple_CNN
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

#parameter
batch_size = 32
num_epochs = 10000
input_shape = (150, 150, 3) #dimensi 3
validation_split = .2
verbose = 1 #tampili detail
num_classes = 101
patience = 50 #learning rate
base_path = 'models/'

#data generator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale= 1./255)

train_generator = train_datagen.flow_from_directory(
    directory = r"food-101.tar/images",
    target_size= (150, 150),
    color_mode = "rgb",
    batch_size = batch_size,
    class_mode = "categorical",
    shuffle = True,
    seed = 42
)
valid_generator = test_datagen.flow_from_directory(
    directory = r"food-101.tar/images",
    target_size= (150, 150),
    color_mode = "rgb",
    batch_size = batch_size,
    class_mode = "categorical",
    shuffle = True,
    seed = 42
)

#model parameter/compilation
model = simple_CNN(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#callback selamo training --ngelog
log_file_path = base_path + 'foodtraininglog.log'
csv_logger = CSVLogger(log_file_path, append=False)
#ngurangi learning rate
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience/4), verbose=1)
trained_models_path = base_path +'simple_CNN'
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1, save_best_only=True)

callbacks = [model_checkpoint, csv_logger,reduce_lr]

#train model
model.fit_generator(generator=train_generator, 
                    validation_data=valid_generator,
                    epochs=num_epochs,
                    verbose=1,
                    callbacks = callbacks,
                    validation_steps = 1,
                    steps_per_epoch=1
)
