import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Check available devices
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    device_name = "GPU"
else:
    device_name = "CPU"

print(f"Using device: {device_name}")

# Create 'kdd_model_checkpoints' directory if it doesn't exist
checkpoint_dir = "kdd_model_checkpoints"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    print(f"Created '{checkpoint_dir}' directory.")

checkpoint_path = checkpoint_dir + "/model_epoch_{epoch:02d}.keras"

checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    save_best_only=False,
    monitor="val_loss",
    mode="min",
    verbose=1,
)


class PrintEpochStats(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}: Loss = {logs['loss']}, Accuracy = {logs['accuracy']}")


print("Loading the NSL-KDD dataset...")

data = pd.read_csv("KDDTrain+.txt", header=None)

# Assigning column names to the dataset
columns = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty_level",
]

data.columns = columns

# Handle categorical columns
categorical_cols = ["protocol_type", "service", "flag"]
label_col = "label"
print("Encoding categorical variables...")
for col in categorical_cols:
    data[col] = LabelEncoder().fit_transform(data[col])

# Convert labels to binary format: 0 for normal, 1 for attack
data["label"] = data["label"].apply(lambda x: 0 if x == "normal" else 1)

# Select features and target
X = data.drop(["label", "difficulty_level"], axis=1)
y = data["label"]

# Preprocessing pipeline (encoding + normalization)
preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), X.columns)])

print("Normalizing the features...")

X = preprocessor.fit_transform(X)

print("Splitting data into training and test sets...")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Define hyperparameters
input_shape = X_train.shape[1]
epochs = 10
batch_size = 32
learning_rate = 0.001

print("Building the neural network model...")

# Build the model
model = models.Sequential(
    [
        layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation="relu"),
        layers.Dense(
            1, activation="sigmoid"
        ),  # Use 'softmax' and modify the layers if doing multi-class classification
    ]
)

print("Compiling the model...")

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Load the latest checkpoint if available
checkpoint_files = sorted(os.listdir(checkpoint_dir))
initial_epoch = 0

if checkpoint_files:
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
    print(f"Resuming from checkpoint: {latest_checkpoint}")
    model.load_weights(latest_checkpoint)
    initial_epoch = int(latest_checkpoint.split("_")[-1].split(".")[0])
    print(f"Resuming from epoch {initial_epoch + 1}...")
else:
    print("No checkpoints found. Starting from scratch.")


print("Training the model...")


# Train the model
model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test, y_test),
    callbacks=[PrintEpochStats(), checkpoint],
    initial_epoch=initial_epoch,
)

print("Model training completed.")
