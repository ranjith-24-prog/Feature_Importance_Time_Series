from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import config

def LSTM_MODEL(X_train,X_test,y_train,y_test):
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.lstm(x)  # Pass through LSTM
            out = self.fc(out[:, -1, :])  # Fully connected on the last time step
            return out

    # Model Parameters
    input_size = config.LSTM["PI_Param"]["input_size"]
    hidden_size = config.LSTM["PI_Param"]["hidden_size"]
    output_size = config.LSTM["PI_Param"]["output_size"]
    num_layers = config.LSTM["PI_Param"]["num_layers"]

    # Initialize the model, loss function, and optimizer
    model = LSTMModel(input_size, hidden_size, output_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ---------- 3. Train the Model ----------
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        print(f"Test Loss: {test_loss.item():.4f}")

    return model

def LSTM_MODEL_LIME(X_train,X_test,y_train,y_test,features_data):
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.lstm(x)  # Pass through LSTM
            out = self.fc(out[:, -1, :])  # Fully connected on the last time step
            return out

    # Model Parameters
    input_size = config.LSTM["LIME_Param"]["input_size"]
    hidden_size = config.LSTM["LIME_Param"]["hidden_size"]
    output_size = y_train.shape[1]
    num_layers = config.LSTM["LIME_Param"]["num_layers"]

    # Initialize the model, loss function, and optimizer
    model = LSTMModel(input_size, hidden_size, output_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ---------- 3. Train the Model ----------
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        print(f"Test Loss: {test_loss.item():.4f}")

    return model