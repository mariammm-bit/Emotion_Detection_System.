"""
This module represents the training pipeline.
It utilizes Fine-tuning on VGG16 and implements parameter tuning 
(EarlyStopping, ReduceLROnPlateau) to achieve the best accuracy.
Note: This script was executed on Kaggle GPUs for performance.
"""
from model import build_model
from preprocess import get_data_generators
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. Get Augmented Data
train_data, val_data = get_data_generators()

# 2. Build the Model Architecture
model = build_model()

# 3. Compile Model with Adam Optimizer
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Define Callbacks for advanced parameter tuning
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=7, 
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=3, 
    min_lr=1e-6
)

# 5. Train the Model (Requires GPU)
if __name__ == "__main__":
    print("Starting Model Training...")
    history = model.fit(
        train_data,
        epochs=40,
        validation_data=val_data,
        callbacks=[early_stop, reduce_lr]
    )
    
    # Save the final trained model
    model.save("Models/emotion_model.h5")
    print("Training Complete. Model Saved.")