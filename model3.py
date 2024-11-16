import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Load datasets
train_data = pd.read_csv('datasets/train.csv', low_memory=False)
test_data = pd.read_csv('datasets/test.csv', low_memory=False)

selected_cols = list(train_data.columns[train_data.columns.str.contains("ImageData.c1c6.summary|ImageData.q1q6.summary")])
selected_cols += [
    "Characteristics.LotSizeSquareFeet",
    "Location.Address.PostalCodePlus4",
    "Property.PropertyType",
    "Structure.BathroomsFull",
    "Structure.BedroomsTotal",
    "Structure.BathroomsHalf",
    "Structure.Cooling",
    "Structure.FireplacesTotal",
    "Structure.LivingArea",
    "Structure.YearBuilt",
    "Structure.GarageSpaces",
    "Structure.Heating",
    "Structure.NewConstructionYN",
    "Location.GIS.Latitude",
    "Location.GIS.Longitude",
    "Listing.Price.ClosePrice",
    "Listing.ListingId"
]

train_data = train_data[selected_cols]
test_data = test_data[[col for col in selected_cols if col != "Listing.Price.ClosePrice"]]

target_col = 'Listing.Price.ClosePrice'
coordinate_cols = ['Location.GIS.Latitude', 'Location.GIS.Longitude']
id_col = 'Listing.ListingId'

X_train = train_data.drop(columns=[target_col] + [id_col])
y_train = train_data[target_col]
X_test = test_data.drop(columns=[id_col])

# 2. Handle NaNs
num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns

X_train[num_cols] = X_train[num_cols].fillna(X_train[num_cols].median())
X_test[num_cols] = X_test[num_cols].fillna(X_train[num_cols].median())

X_train[cat_cols] = X_train[cat_cols].infer_objects().fillna(X_train[cat_cols].mode().iloc[0])
X_test[cat_cols] = X_test[cat_cols].infer_objects().fillna(X_train[cat_cols].mode().iloc[0])

# 3. Preprocess the data
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols),
        ('coords', StandardScaler(), coordinate_cols)  # Include coordinates in preprocessing
    ]
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Log-transform the target variable
y_train = np.log1p(y_train)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

# 4. PyTorch Dataset
class HouseholdDataset(Dataset):
    def __init__(self, ids, features, labels=None):
        self.features = torch.tensor(features.toarray(), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        self.ids = ids

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx], self.ids[idx]
        else:
            return self.features[idx]

train_dataset = HouseholdDataset(train_data[id_col].to_numpy(), X_train, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 5. Improved PyTorch Model
class PricePredictor(nn.Module):
    def __init__(self, input_dim):
        super(PricePredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self._initialize_weights()

    def forward(self, x):
        return self.fc(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

# Initialize the model
input_dim = X_train.shape[1]
model = PricePredictor(input_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

# 6. Training Loop
actual_prices = []
predicted_prices = []
idss = []
epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for features, labels, ids in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if epoch == epochs-1:
            idss.extend(list(ids))
            predicted_prices.extend(outputs.squeeze().tolist())
            actual_prices.extend(labels.tolist())
    scheduler.step(running_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

print("Training finished!")

# Add this after the training loop
# Create a DataFrame with Actual Prices, Predicted Prices, and IDs
results_df = pd.DataFrame({
    'ListingID': idss,  # IDs from the dataset
    'ActualPrice': np.expm1(actual_prices),  # Reverse log-transform for actual prices
    'PredictedPrice': np.expm1(predicted_prices)  # Reverse log-transform for predicted prices
})

# Save results to CSV
results_df.to_csv('model3_train.csv', index=False)
print("Results saved to 'model3_train.csv'")

torch.save(model, "model.pth")  # Save the entire model

# Make predictions for the test set
model.eval()
test_dataset = HouseholdDataset(test_data[id_col].to_numpy(), X_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

predictions = []
with torch.no_grad():
    for features in test_loader:
        outputs = model(features)
        predictions.extend(outputs.squeeze().tolist())

# Inverse the log transformation of predictions
predicted_prices = np.expm1(predictions)
# Add coordinates, ListingId, and predicted prices to the output
output_df = test_data[coordinate_cols + [id_col]].copy()
output_df['predicted_price'] = predicted_prices

# Save predictions to CSV
output_df.to_csv('model3_test.csv', index=False)
print("Predictions saved to 'model3_test.csv'")

