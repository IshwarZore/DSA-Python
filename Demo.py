# ========================== NumPy ==========================
import numpy as np
arr = np.array([1, 2, 3]); matrix = np.array([[1, 2], [3, 4]])
arr.shape; arr.reshape(3, 1)
np.mean(arr); np.std(arr); np.sum(arr)
np.dot(arr, arr)
np.arange(0, 10, 2); np.linspace(0, 1, 5)
np.random.rand(2, 3)

# ========================== Pandas ==========================
import pandas as pd
df = pd.read_csv('file.csv'); df.to_csv('out.csv', index=False)
df.head(); df.info(); df.describe(); df.columns; df.shape
df['col']; df[['col1', 'col2']]; df.iloc[0]; df.loc[0, 'col']
df[df['Age'] > 20]; df.sort_values('Age')
df.groupby('target').mean(); df.isnull().sum()
df.fillna(0); df.dropna(); df['col'].value_counts()

# ========================== Matplotlib ==========================
import matplotlib.pyplot as plt
x = [1, 2, 3]; y = [10, 20, 15]
plt.plot(x, y); plt.bar(x, y); plt.scatter(x, y); plt.hist(y)
plt.xlabel("X"); plt.ylabel("Y"); plt.title("Title"); plt.grid(True)
plt.show()

# ========================== Seaborn ==========================
import seaborn as sns
df = sns.load_dataset('iris')
sns.histplot(df['sepal_length']); sns.boxplot(x='species', y='sepal_length', data=df)
sns.violinplot(x='species', y='sepal_length', data=df)
sns.pairplot(df, hue='species')
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

# ========================== Scikit-learn Basics ==========================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

# ========================== Feature Scaling + Encoding ==========================
from sklearn.preprocessing import StandardScaler, LabelEncoder
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ========================== Stratified K-Fold ==========================
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5)
model = LogisticRegression()

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Fold accuracy:", accuracy_score(y_test, preds))

# ========================== Pipeline + GridSearchCV ==========================
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])

params = {
    'clf__C': [0.1, 1, 10]
}

grid = GridSearchCV(pipeline, param_grid=params, cv=5)
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.score(X_test, y_test))

# ========================== PCA (Dimensionality Reduction) ==========================
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ========================== Keras Neural Network ==========================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 3-class classification
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_scaled, y_encoded, epochs=10, batch_size=16, validation_split=0.2)

# ========================== PyTorch Neural Network ==========================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.factorize()[0], dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(X.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")