import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Load the datasets with proper dtype handling
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
    "Listing.ListingId"  # Add ListingId for the final output
]

train_data = train_data[selected_cols]
test_data = test_data[[col for col in selected_cols if col != "Listing.Price.ClosePrice"]]

#print(train_data[train_data['Listing.ListingId'] == "mrd11827406"])
#exit()

print("Train data snapshot:\n", train_data.head())
print("Test data snapshot:\n", test_data.head())

# Separate features and target in the train dataset
target_col = 'Listing.Price.ClosePrice'
coordinate_cols = ['Location.GIS.Latitude', 'Location.GIS.Longitude']
# Tak
id_col = 'Listing.ListingId'
X_train = train_data.drop(columns=[target_col] + coordinate_cols + [id_col])
y_train = train_data[target_col]
X_test = test_data.drop(columns=coordinate_cols + [id_col])

# 2. Handle NaNs
# Fill NaNs in numerical columns with the median
num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
X_train[num_cols] = X_train[num_cols].fillna(X_train[num_cols].median())
X_test[num_cols] = X_test[num_cols].fillna(X_train[num_cols].median())

# Fill NaNs in categorical columns with the mode
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
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
        ('cat', categorical_transformer, cat_cols)
    ]
)

# Apply preprocessing
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Log-transform the target variable
y_train = np.log1p(y_train)

# Convert y to PyTorch tensors
y_train_tensor = y_train.values if isinstance(y_train.values, torch.Tensor) else torch.tensor(y_train.values, dtype=torch.float32)

# 4. Define a PyTorch Dataset
class HouseholdDataset(Dataset):
    def __init__(self, ids, features, labels=None):
        # Handle features
        self.features = features.toarray() if not isinstance(features, torch.Tensor) else features.clone().detach()
        self.features = torch.tensor(self.features, dtype=torch.float32)  # Convert to tensor if not already

        # Handle labels
        if labels is not None:
            self.labels = labels.clone().detach() if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.float32)
        else:
            self.labels = None

        # Store IDs
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


# 5. Build the PyTorch model
class PricePredictor(nn.Module):
    def __init__(self, input_dim):
        super(PricePredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
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
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

# Initialize the model
input_dim = X_train.shape[1]
model = PricePredictor(input_dim)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Train the model
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
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

print("Training finished!")


# Create a DataFrame with the results
results = pd.DataFrame({
    'ID': idss,
    'Actual Price': np.expm1(actual_prices),
    'Predicted Price': np.expm1(predicted_prices)
})


# Save the DataFrame to a CSV file
results.to_csv('model1_train.csv', index=False)

print("Predictions saved to model1_train.csv")


# 7. Make predictions for the test set
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
output_df.to_csv('model1_test.csv', index=False)
print("Predictions saved to 'model1_test.csv'")
