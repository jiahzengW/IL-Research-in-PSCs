import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# tabular data has 21 features and image data is grayscale with size 64x64
tabular_feature_size = 21
image_height, image_width, image_channels = 64, 64, 1
output_size = 1  # Adjust for your specific problem

# Define the dual-modal model with reduced complexity
tabular_input = Input(shape=(tabular_feature_size,), name='tabular_input')
tabular_branch = Dense(64, activation='relu')(tabular_input)
tabular_branch = Dropout(0.4)(tabular_branch)

image_input = Input(shape=(image_height, image_width, image_channels), name='image_input')
image_branch = Conv2D(32, (3, 3), activation='relu')(image_input)
image_branch = MaxPooling2D((2, 2))(image_branch)
image_branch = Conv2D(32, (3, 3), activation='relu')(image_branch)
image_branch = MaxPooling2D((2, 2))(image_branch)

image_branch = Flatten()(image_branch)
image_branch = Dropout(0.4)(image_branch)

concatenated = Concatenate(name='concatenated')([tabular_branch, image_branch])

dense_layer = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(concatenated)
dense_layer = BatchNormalization()(dense_layer)
dense_layer = Dropout(0.3)(dense_layer)

output_layer = Dense(output_size, activation='linear', name='output')(dense_layer)

dual_modal_model = Model(inputs=[tabular_input, image_input], outputs=output_layer)
dual_modal_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

dual_modal_model.summary()

######### Data Preparation: #################

# Load your data here
X_tabular = np.load()
X_imageF = np.load()
y = np.load()
print("Loaded X_tabular shape:", X_tabular.shape)
print("Loaded X_imageF shape:", X_imageF.shape)
print("Loaded y shape:", y.shape)

# Train-test split
tabular_train_data, tabular_test_data, image_train_data, image_test_data, train_labels, test_labels = train_test_split(
    X_tabular, X_imageF, y, test_size=0.2, random_state=42
)

# Display some information about the data
print("Tabular data shape:", tabular_train_data.shape)
print("Image data shape:", image_train_data.shape)
print("Labels shape:", train_labels.shape)

# Training the model with early stopping
early_stopping = EarlyStopping(patience=20, restore_best_weights=True)

# Train the model
history = dual_modal_model.fit(
    [tabular_train_data, image_train_data],
    train_labels,
    epochs=100,  # Adjust the number of epochs
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)


##########Using Only Image Data:###################
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
# Assuming image data is grayscale with size 64x64
image_height, image_width, image_channels = 64, 64, 1
output_size = 1

# Define the image-only model
image_input = Input(shape=(image_height, image_width, image_channels), name='image_input')
image_branch = Conv2D(16, (3, 3), activation='relu')(image_input)
image_branch = MaxPooling2D((2, 2))(image_branch)
image_branch = Flatten()(image_branch)
image_branch = Dropout(0.3)(image_branch)

dense_layer = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005))(image_branch)
dense_layer = BatchNormalization()(dense_layer)
dense_layer = Dropout(0.3)(dense_layer)

output_layer = Dense(output_size, activation='linear', name='output')(dense_layer)

image_only_model = Model(inputs=image_input, outputs=output_layer)
image_only_model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])

image_only_model.summary()


#############Using Only Tabular Data:################
# Code for tabular-only model and training
tabular_feature_size = 21
output_size = 1

# Define the tabular-only model
tabular_input = Input(shape=(tabular_feature_size,), name='tabular_input')
tabular_branch = Dense(32, activation='relu')(tabular_input)
tabular_branch = Dropout(0.2)(tabular_branch)

dense_layer = Dense(64, activation='relu',kernel_regularizer=l2(0.01))(tabular_branch)
dense_layer = BatchNormalization()(dense_layer)
dense_layer = Dropout(0.2)(dense_layer)

output_layer = Dense(output_size, activation='linear', name='output')(dense_layer)

tabular_only_model = Model(inputs=tabular_input, outputs=output_layer)
tabular_only_model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])

tabular_only_model.summary()

# Train-test split
tabular_train_data, tabular_test_data, train_labels, test_labels = train_test_split(
    X_tabular, y, test_size=0.2, random_state=42
)

# Training the model with early stopping
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

# Training the tabular-only model
history_tabular = tabular_only_model.fit(
    tabular_train_data,
    train_labels,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)
