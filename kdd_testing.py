import tensorflow as tf
from tensorflow.keras import models
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Check available devices
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    device_name = "GPU"
else:
    device_name = "CPU"

print(f"Using device: {device_name}")

print("Loading the NSL-KDD dataset...")

data = pd.read_csv("KDDTrain+.txt", header=None)

checkpoint_dir = "kdd_model_checkpoints"

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

# Load the latest checkpoint if available
checkpoint_files = sorted(os.listdir(checkpoint_dir))
final_epoch = 0

if checkpoint_files:
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
    final_epoch = int(latest_checkpoint.split("_")[-1].split(".")[0])
    print(f"Loading the final model from epoch: {final_epoch}...")
    model = models.load_model(latest_checkpoint)
else:
    raise FileNotFoundError("No checkpoints found. Consider training the model first.")

print("Evaluating the model...")

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy*100:.2f}%")

print("Model evaluation completed. Generating classification report...")

# Generate a classification report
y_pred = (model.predict(X_test) > 0.5).astype("int32")
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

# Save results to a file
results_path = "kdd_final_test_results.txt"
with open(results_path, "w") as f:
    f.write(f"Final Model Evaluation Results:\n\n\n")
    f.write(f"Number of training epochs: {final_epoch}\n\n")
    f.write(f"Loss: {loss:.2f}\n\n")
    f.write(f"Accuracy: {accuracy*100:.2f}%\n\n")
    f.write(f"Confusion Matrix:\n{matrix}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print(f"Final results saved to {results_path}")
