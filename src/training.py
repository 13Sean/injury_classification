
import torch
import numpy as np
from .feature_extraction import frequency_domain_augmentation


def train_epoch(model, X_train, y_train, optimizer, loss_functions, loss_weights, 
                batch_size, device, feature_dim, gradient_clip=1.0):

    model.train()
    train_loss = 0.0
    
    for i in range(0, len(X_train), batch_size):
        inputs = X_train[i:i + batch_size].numpy()
        labels = y_train[i:i + batch_size]

        # 应用频域数据增强 (50%概率)
        if np.random.rand() > 0.5:
            inputs = frequency_domain_augmentation(inputs)

        inputs = torch.tensor(inputs, dtype=torch.float32).view(-1, 1, feature_dim).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)


        total_loss = 0.0
        for idx, (criterion, weight) in enumerate(zip(loss_functions, loss_weights)):
            if idx == 4:
                loss = criterion(outputs, labels, inputs)
            else:
                loss = criterion(outputs, labels)
            total_loss += weight * loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()

        train_loss += total_loss.item() * inputs.size(0)

    return train_loss / len(X_train)


def validate(model, X_test, y_test, loss_functions, loss_weights, batch_size, device):

    model.eval()
    valid_loss = 0.0
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            inputs = X_test[i:i + batch_size].to(device)
            labels = y_test[i:i + batch_size].to(device)
            outputs = model(inputs)


            total_loss = 0.0
            for idx, (criterion, weight) in enumerate(zip(loss_functions, loss_weights)):
                if idx == 4:
                    loss = criterion(outputs, labels, inputs)
                else:
                    loss = criterion(outputs, labels)
                total_loss += weight * loss

            valid_loss += total_loss.item() * inputs.size(0)
            _, pred = torch.max(outputs, 1)
            predictions.extend(pred.cpu().numpy())

    return valid_loss / len(X_test), predictions


def train_model(model, X_train, X_test, y_train, y_test, optimizer, scheduler,
                loss_functions, loss_weights, num_epochs, batch_size, device, feature_dim):

    train_losses = []
    valid_losses = []
    final_predictions = []

    for epoch in range(num_epochs):

        train_loss = train_epoch(
            model, X_train, y_train, optimizer, loss_functions, loss_weights,
            batch_size, device, feature_dim
        )
        train_losses.append(train_loss)


        valid_loss, predictions = validate(
            model, X_test, y_test, loss_functions, loss_weights, batch_size, device
        )
        valid_losses.append(valid_loss)
        
        if epoch == num_epochs - 1:
            final_predictions = predictions
        
        scheduler.step(valid_loss)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], '
            f'Train Loss: {train_loss:.4f}, '
            f'Valid Loss: {valid_loss:.4f}'
        )

    return train_losses, valid_losses, final_predictions
