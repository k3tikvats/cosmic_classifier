import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import joblib
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')


train_file = "cosmic-classifier-cogni25\data\cosmicclassifierTraining.csv"
data = pd.read_csv(train_file)
print("Initial data shape:", data.shape)


expected_columns = ["Atmospheric_Density", "Surface_Temperature", "Gravity",
                    "Water_Content", "Mineral_Abundance", "Orbital_Period",
                    "Proximity_to_Star", "Magnetic_Field_Strength", "Radiation_Levels",
                    "Atmospheric_Composition_Index", "Planet_Class"]
if list(data.columns) != expected_columns:
    data.columns = expected_columns

# Clean the data
data = data.dropna()
print("Data shape after dropping missing values:", data.shape)

def valid_label(x):
    if isinstance(x, (int, float)):
        return x >= 0
    return True

data_clean = data[data["Planet_Class"].apply(valid_label)]
print("Data shape after filtering invalid labels:", data_clean.shape)


class_counts = data_clean['Planet_Class'].value_counts().sort_index()
print("Number of rows per class")
for class_value, count in class_counts.items():
    print(f"Class {class_value}: {count}")

class_names = {
    0: "Bewohnbar",        # Habitable
    1: "Terraformierbar",  # Terraformable
    2: "Rohstoffreich",    # Resource-rich
    3: "Wissenschaftlich", # Scientific
    4: "Gasriese",         # Gas giant
    5: "Wüstenplanet",     # Desert planet
    6: "Eiswelt",          # Ice world
    7: "Toxischetmosäre",  # Toxic atmosphere
    8: "Hohestrahlung",    # High radiation
    9: "Toterahswelt"      # Dead world
}

# Extract features
features = data_clean.drop("Planet_Class", axis=1)


non_numeric_cols = features.select_dtypes(exclude=[np.number]).columns
if len(non_numeric_cols) > 0:
    print("Extracting numeric values from non-numeric columns:", non_numeric_cols.tolist())
    for col in non_numeric_cols:
        features[col] = features[col].str.extract('(\d+)').astype(float)

# feature engineering - particularly for distinguishing between classes 4 and 9
print("Adding engineered features...")
# Radiation-related interactions (important for Class 9 - High Radiation)
features['radiation_temp_interaction'] = features['Radiation_Levels'] * features['Surface_Temperature']
features['radiation_atmosphere_interaction'] = features['Radiation_Levels'] * features['Atmospheric_Composition_Index']
features['radiation_magnetic_ratio'] = features['Radiation_Levels'] / (features['Magnetic_Field_Strength'] + 1e-5)

# Gas giant related features (for Class 4)
features['density_gravity_ratio'] = features['Atmospheric_Density'] / (features['Gravity'] + 1e-5)
features['orbital_proximity_ratio'] = features['Orbital_Period'] / (features['Proximity_to_Star'] + 1e-5)

# Other potentially useful feature combinations
features['water_temp_interaction'] = features['Water_Content'] * features['Surface_Temperature']
features['mineral_gravity_product'] = features['Mineral_Abundance'] * features['Gravity']
features['habitability_index'] = (features['Atmospheric_Composition_Index'] * 0.4 + 
                                 (1 - abs(features['Surface_Temperature'])) * 0.3 +
                                 (1 - features['Radiation_Levels']) * 0.3)

print("Feature set shape after engineering:", features.shape)

# Feature importance analysis to understand what differentiates classes 4 and 9
X = features.values
y = data_clean["Planet_Class"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Encoded classes:", le.classes_)

# Calculate feature importance using mutual information
feature_importance = mutual_info_classif(X, y_encoded)
importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': feature_importance})
importance_df = importance_df.sort_values('Importance', ascending=False)
print("\nFeature importance:")
print(importance_df.head(10))

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create more balanced datasets with focused augmentation for confused classes
print("\nPerforming targeted data augmentation for frequently confused classes...")
# Split data
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Identify samples from most confused classes (4 & 9)
class4_indices = np.where(y_train == 4)[0]
class9_indices = np.where(y_train == 9)[0]

# Create synthetic samples with small perturbations for these classes
aug_samples = []
aug_labels = []

for idx in np.concatenate([class4_indices, class9_indices]):
    for _ in range(2):  # Create 2 synthetic samples per original
        noise = np.random.normal(0, 0.05, X_train.shape[1])  # Small perturbations
        perturbed = X_train[idx] + noise
        aug_samples.append(perturbed)
        aug_labels.append(y_train[idx])

# Add augmented samples to training data
X_train_aug = np.vstack([X_train, np.array(aug_samples)])
y_train_aug = np.concatenate([y_train, np.array(aug_labels)])
print(f"Training data shape after augmentation: {X_train_aug.shape}")

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_aug, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_aug, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Create DataLoaders
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Improved MLP model
class EnhancedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.3):
        super(EnhancedMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Create multiple hidden layers with different dimensions
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            
            # Use different activation functions in different layers
            if i % 2 == 0:
                layers.append(nn.SiLU(0.1))
            else:
                layers.append(nn.GELU())
                
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
        # Apply custom weight initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        return self.model(x)

# Define model parameters
input_dim = X_train_aug.shape[1]
hidden_dims = [256,128, 96, 64, 32]  # Deeper network with varying layer sizes
output_dim = len(le.classes_)
dropout_rate =0.25 # Custom dropout layer

# Create the model
model = EnhancedMLP(input_dim, hidden_dims, output_dim, dropout_rate)
print(model)

# Calculate class weights for weighted loss function
class_samples = np.bincount(y_train)
total_samples = len(y_train)
class_weights = torch.tensor(
    [total_samples / (len(class_samples) * count) for count in class_samples],
    dtype=torch.float32
)
print("Class weights:", class_weights)

# Define loss function with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Use AdamW optimizer with weight decay for better regularization
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

# Training parameters
num_epochs = 100
train_losses = []
val_losses = []
val_accuracies = []
best_val_acc = 0
best_model_state = None
patience = 10
patience_counter = 0

print("\nTraining the model...")
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(batch_y.cpu().numpy())
            
    epoch_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    
    val_acc = accuracy_score(all_true, all_preds)
    val_accuracies.append(val_acc)
    
    # Update learning rate based on validation loss
    scheduler.step(epoch_val_loss)
    
    # Print metrics
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        patience_counter = 0
        
        # Detailed metrics for best model
        print("\nIntermediate Classification Report:")
        target_names = [f"{i}: {class_names[i]}" for i in range(len(class_names))]
        print(classification_report(all_true, all_preds, target_names=target_names))
        
        # Focus on problematic classes
        class4_true = np.array(all_true) == 4
        class4_pred = np.array(all_preds) == 4
        class9_true = np.array(all_true) == 9
        class9_pred = np.array(all_preds) == 9
        
        print(f"Class 4 (Gasriese) accuracy: {np.sum(class4_true & class4_pred) / np.sum(class4_true):.4f}")
        print(f"Class 9 (Hohestrahlung) accuracy: {np.sum(class9_true & class9_pred) / np.sum(class9_true):.4f}")
    else:
        patience_counter += 1
    
    # Early stopping
    # if patience_counter >= patience:
    #     print(f"Early stopping triggered after {epoch+1} epochs")
    #     break

# Load best model for evaluation
model.load_state_dict(best_model_state)

# Plot training curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.axhline(y=best_val_acc, color='r', linestyle='--', label=f'Best Acc: {best_val_acc:.4f}')
plt.legend()
plt.show()

# Final evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_val_tensor)
    probs = nn.Softmax(dim=1)(outputs)
    preds = torch.argmax(outputs, dim=1)
    pred_probs, _ = torch.max(probs, dim=1)
    all_preds = preds.cpu().numpy()
    pred_probs = pred_probs.cpu().numpy()

final_acc = accuracy_score(y_val, all_preds)
print("Final Validation Accuracy: {:.2f}%".format(final_acc * 100))

# Generate detailed classification report
target_names = [f"{i}: {class_names[i]}" for i in range(len(class_names))]
print("\nClassification Report:")
print(classification_report(y_val, all_preds, target_names=target_names))

# Create and display confusion matrix
conf_matrix = confusion_matrix(y_val, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()
plt.show()

# Visualize embeddings with t-SNE
with torch.no_grad():
    # Get embeddings from the penultimate layer
    feature_extractor = nn.Sequential(*list(model.model.children())[:-1])
    embeddings = feature_extractor(X_val_tensor).cpu().numpy()

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot with true labels
plt.figure(figsize=(14, 12))
# Create a custom colormap
colors = plt.cm.viridis(np.linspace(0, 1, len(class_names)))
for i, color in enumerate(colors):
    mask = y_val == i
    plt.scatter(
        embeddings_2d[mask, 0], 
        embeddings_2d[mask, 1],
        c=[color], 
        label=class_names[i],
        alpha=0.7,
        s=50
    )
plt.title("t-SNE of Validation Embeddings (True Labels)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot with predicted labels and confidence
plt.figure(figsize=(14, 12))
scatter = plt.scatter(
    embeddings_2d[:, 0], 
    embeddings_2d[:, 1], 
    c=all_preds, 
    cmap='viridis', 
    alpha=0.7,
    s=50*pred_probs  # Size points by confidence
)
plt.colorbar(scatter)
plt.title("t-SNE of Validation Embeddings (Predicted Labels, size = confidence)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.tight_layout()
plt.show()

# Save the model, preprocessing objects, and feature engineering steps
model_info = {
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'label_encoder': le,
    'input_dim': input_dim,
    'hidden_dims': hidden_dims,
    'output_dim': output_dim,
    'dropout_rate': dropout_rate,
    'feature_names': list(features.columns),
    'class_names': class_names
}

torch.save(model_info, "improved_cosmic_classifier_model.pth")
print("Model and preprocessing objects saved.")

# Bonus: Create a prediction function for new data
def predict_planet_class(model, new_data, model_info):
    """
    Predict planet class for new data.
    
    Args:
        model: PyTorch model
        new_data: DataFrame with the same columns as the training data
        model_info: Dictionary with model metadata
    
    Returns:
        Tuple of (predicted_class, class_name, confidence)
    """
    # Ensure we have the same features (including engineered ones)
    feature_names = model_info['feature_names']
    required_raw_features = ['Atmospheric_Density', 'Surface_Temperature', 'Gravity',
                          'Water_Content', 'Mineral_Abundance', 'Orbital_Period',
                          'Proximity_to_Star', 'Magnetic_Field_Strength', 'Radiation_Levels',
                          'Atmospheric_Composition_Index']
    
    # Check if we have all required features
    if not all(col in new_data.columns for col in required_raw_features):
        raise ValueError(f"Input data must have these columns: {required_raw_features}")
    
    # Create engineered features
    df = new_data.copy()
    df['radiation_temp_interaction'] = df['Radiation_Levels'] * df['Surface_Temperature']
    df['radiation_atmosphere_interaction'] = df['Radiation_Levels'] * df['Atmospheric_Composition_Index']
    df['radiation_magnetic_ratio'] = df['Radiation_Levels'] / (df['Magnetic_Field_Strength'] + 1e-5)
    df['density_gravity_ratio'] = df['Atmospheric_Density'] / (df['Gravity'] + 1e-5)
    df['orbital_proximity_ratio'] = df['Orbital_Period'] / (df['Proximity_to_Star'] + 1e-5)
    df['water_temp_interaction'] = df['Water_Content'] * df['Surface_Temperature']
    df['mineral_gravity_product'] = df['Mineral_Abundance'] * df['Gravity']
    df['habitability_index'] = (df['Atmospheric_Composition_Index'] * 0.4 + 
                              (1 - abs(df['Surface_Temperature'])) * 0.3 +
                              (1 - df['Radiation_Levels']) * 0.3)
    
    # Scale the data
    X = model_info['scaler'].transform(df[feature_names].values)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.argmax(outputs, dim=1)
        confidence, _ = torch.max(probs, dim=1)
    
    predicted_class = preds.item()
    class_name = model_info['class_names'][predicted_class]
    
    return predicted_class, class_name, confidence.item()

print("\nExample usage of the prediction function:")
print("predict_planet_class(model, new_data, model_info)")