
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.optim as optim
from config import *
from src.data_loader import load_data, normalize_features, split_data, convert_to_tensor
from src.feature_extraction import (apply_fft_to_samples, normalize_fft_features,
                                     concatenate_features, apply_pca)
from src.models import HybridModel
from src.losses import initialize_losses
from src.training import train_model
from src.evaluation import (calculate_metrics, calculate_roc_curves,
                             plot_loss_curve, plot_micro_roc, plot_class_roc, print_results)


def main():


    

    os.makedirs(RESULTS_DIR, exist_ok=True)

    

    features, target = load_data(DATA_PATH, FEATURE_NAMES)


    features, scaler = normalize_features(features)
    

    fft_features = apply_fft_to_samples(features)
    fft_features, fft_scaler = normalize_fft_features(fft_features)
    features = concatenate_features(features, fft_features)
    features, pca = apply_pca(features, variance_ratio=0.95)

    X_train, X_test, y_train, y_test = split_data(
        features, target, TEST_SIZE, RANDOM_STATE
    )
    X_train, X_test, y_train, y_test = convert_to_tensor(
        X_train, X_test, y_train, y_test, features.shape[1]
    )

    model = HybridModel(
        input_size=features.shape[1],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        rank=RANK
    ).to(DEVICE)
    

    loss_functions, loss_weights = initialize_losses()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    train_losses, valid_losses, predictions = train_model(
        model, X_train, X_test, y_train, y_test,
        optimizer, scheduler, loss_functions, loss_weights,
        NUM_EPOCHS, BATCH_SIZE, DEVICE, features.shape[1]
    )
    

    
    y_true = y_test.numpy()
    y_pred = np.array(predictions)
    
    metrics = calculate_metrics(y_true, y_pred, NUM_CLASSES)
    fpr, tpr, roc_auc = calculate_roc_curves(y_true, y_pred, NUM_CLASSES)
    

    plot_loss_curve(train_losses, valid_losses, NUM_EPOCHS, LOSS_CURVE_PATH)
    plot_micro_roc(fpr, tpr, roc_auc, MICRO_ROC_PATH)
    plot_class_roc(fpr, tpr, roc_auc, CLASS_NAMES, NUM_CLASSES, CLASS_ROC_PATH)
    

    print_results(metrics, roc_auc, features.shape[1], CLASS_NAMES)
    

    model_path = f'{RESULTS_DIR}/model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\model saved: {model_path}")
    



if __name__ == "__main__":
    main()
