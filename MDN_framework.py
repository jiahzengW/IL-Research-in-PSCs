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

# tabular data has nN features and image data is grayscale with size 64x64
tabular_feature_size = nN
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
tabular_feature_size = nN
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


# Evaluate the model
train_loss, train_mae = dual_modal_model.evaluate([tabular_train_data, image_train_data], train_labels)
test_loss, test_mae = dual_modal_model.evaluate([tabular_test_data, image_test_data], test_labels)

# Predictions for training and testing sets
y_train_pred = dual_modal_model.predict([tabular_train_data, image_train_data])
y_test_pred = dual_modal_model.predict([tabular_test_data, image_test_data])

# Calculate R2 scores
train_r2 = r2_score(train_labels, y_train_pred)
test_r2 = r2_score(test_labels, y_test_pred)

print(f'Training Mean Absolute Error: {train_mae}')
print(f'Test Mean Absolute Error: {test_mae}')
print(f'Training R2 Score: {train_r2:.3f}')
print(f'Test R2 Score: {test_r2:.3f}')

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Training and Validation Loss Over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)

def create_custom_scatter_plot(y_true, y_pred, model_name, color, marker, r2_score):
    alpha_value = 0.6  # Adjust alpha for transparency
    plt.scatter(y_true, y_pred, s=50, c=color, marker=marker, alpha=alpha_value, edgecolor='None', linewidth=0.5)
    plt.title(f'{model_name} Model', fontsize=10, fontweight='bold')
    plt.xlabel('Actual PCE(%)', fontsize=12)
    plt.ylabel('Predicted PCE(%)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

# Function to draw combined scatter plots
def draw_combined_scatter_plot(model, X_tabular_train, X_image_train, y_train, X_tabular_test, X_image_test, y_test, model_name, train_sample_size=200, test_sample_size=50):
    # Sample data points
    idx_train = np.random.choice(len(y_train), min(train_sample_size, len(y_train)), replace=False)
    idx_test = np.random.choice(len(y_test), min(test_sample_size, len(y_test)), replace=False)

    # Predictions
    y_train_pred = model.predict([X_tabular_train[idx_train], X_image_train[idx_train]]).flatten()
    y_test_pred = model.predict([X_tabular_test[idx_test], X_image_test[idx_test]]).flatten()

    # Calculate R2 scores
    train_r2 = r2_score(y_train[idx_train], y_train_pred)
    test_r2 = r2_score(y_test[idx_test], y_test_pred)

    # Plot
    plt.figure(figsize=(6, 6))
    create_custom_scatter_plot(y_train[idx_train], y_train_pred, model_name, 'lightblue', 'o', train_r2)
    create_custom_scatter_plot(y_test[idx_test], y_test_pred, model_name, 'lightcoral', '^', test_r2)

    # Adding legend
    train_marker = mlines.Line2D([], [], color='lightblue', marker='o', linestyle='None', markersize=6, label='Train Data')
    test_marker = mlines.Line2D([], [], color='lightcoral', marker='^', linestyle='None', markersize=6, label='Test Data')
    plt.legend(handles=[train_marker, test_marker], loc='upper left', fontsize=9)

    plt.show()

# Call the function for combined scatter plot
draw_combined_scatter_plot(dual_modal_model, tabular_train_data, image_train_data, train_labels, tabular_test_data, image_test_data, test_labels, "Dual Modal Model")





# Train-test split
image_train_data, image_test_data, train_labels, test_labels = train_test_split(
    X_imageF, y, test_size=0.2, random_state=42
)

# Training the model with early stopping
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

history_image = image_only_model.fit(
    image_train_data,
    train_labels,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)

#############Using Only Image Data:################
# Evaluation
train_loss, train_mae = image_only_model.evaluate(image_train_data, train_labels)
train_predictions = image_only_model.predict(image_train_data)
train_r2 = r2_score(train_labels, train_predictions)

test_loss, test_mae = image_only_model.evaluate(image_test_data, test_labels)
test_predictions = image_only_model.predict(image_test_data)
test_r2 = r2_score(test_labels, test_predictions)

print(f'Training Mean Absolute Error: {train_mae}, R2: {train_r2}')
print(f'Test Mean Absolute Error: {test_mae}, R2: {test_r2}')

# Plotting the training and validation loss over epochs
plt.plot(history_image.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history_image.history['val_loss'], label='Validation Loss', linewidth=2)

# Annotating R2 scores
plt.text(0.5 * len(history_image.history['loss']), np.min(history_image.history['loss']), f'Train R2: {train_r2:.3f}', fontsize=12)
plt.text(0.5 * len(history_image.history['val_loss']), np.min(history_image.history['val_loss']), f'Test R2: {test_r2:.3f}', fontsize=12)

plt.title('Training and Validation Loss Over Epochs (Image-Only Model)', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)


#############Using Only Tabular Data:################
# Evaluation
train_loss, train_mae = tabular_only_model.evaluate(tabular_train_data, train_labels)
print(f'Training Mean Absolute Error: {train_mae}')

test_loss, test_mae = tabular_only_model.evaluate(tabular_test_data, test_labels)
print(f'Test Mean Absolute Error: {test_mae}')

# Plotting the training and validation loss over epochs for tabular-only model
plt.figure(figsize=(10, 6))
plt.plot(history_tabular.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history_tabular.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Training and Validation Loss Over Epochs (Tabular-Only Model)', fontsize=16)
