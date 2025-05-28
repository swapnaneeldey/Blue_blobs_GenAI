import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import AdamW
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os


class ModelFineTuner:
    def __init__(self, model, base_model):
        """
        Initialize the ModelFineTuner.
        
        Args:
            model: your keras Model (head-trained) with a ResNet50V2 backbone
            base_model: the actual backbone model object (not just the name)
        """
        self.model = model
        self.base_model = base_model
        self.history = None
    
    def plot_training_history(self, history=None):
        """
        Plot the training and validation metrics.
        
        Args:
            history: Training history. If None, uses self.history
        """
        if history is None:
            history = self.history
            
        if history is None:
            print("No training history available to plot.")
            return
            
        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

        plt.tight_layout()
        plt.show()
        
    def plot_confusion_matrix(self, X_data, y_true, class_names=None):
        """
        Plot confusion matrix for model predictions.
        
        Args:
            X_data: Input data
            y_true: Ground truth labels
            class_names: Optional list of class names
        """
        y_pred = (self.model.predict(X_data) > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        plt.figure(figsize=(8, 8))
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))

    def fine_tune(
        self,
        X_train, y_train,
        X_val, y_val,
        num_unfreeze_layers: int = 3,
        learning_rate: float = 1e-6,
        epochs: int = 20,
        batch_size: int = 32,
        es_patience: int = 5,
        rlr_patience: int = 4,
        gradual_unfreeze: bool = False,
        early_stop = str ("val_recall_gal"),
    ):
        """
        Fine-tune the model using separate training and validation sets.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            num_unfreeze_layers: Number of layers to unfreeze from the backbone
            learning_rate: Initial learning rate for fine-tuning
            epochs: Maximum number of epochs to train
            batch_size: Batch size for training
            es_patience: Early stopping patience based on validation recall
            rlr_patience: Learning rate reduction patience based on validation loss
            gradual_unfreeze: Whether to gradually unfreeze layers during training
            
        Returns:
            Training history
        """
        # 1) Compute class weights from training data only
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight = dict(zip(classes, weights))
        # (optional tweak - can be adjusted based on your problem)
        if 1 in class_weight:
            class_weight[1] *= 1.0
        print(f"Class weights for training dataset: {class_weight}")

        # 2) Unfreeze backbone layers
        if gradual_unfreeze:
            # Start with all layers frozen
            self.base_model.trainable = True
            for layer in self.base_model.layers:
                layer.trainable = False
        else:
            # Standard approach: unfreeze last n layers
            self.base_model.trainable = True
            for layer in self.base_model.layers[:-num_unfreeze_layers]:
                layer.trainable = False
                
        # Print which layers are trainable for verification
        trainable_layers = [layer.name for layer in self.base_model.layers if layer.trainable]
        print(f"Trainable backbone layers: {trainable_layers}")

        
        self.model.compile(
            optimizer=AdamW(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                keras.metrics.Precision(name="precision_gal"),
                keras.metrics.Recall(name="recall_gal"),
            ]
        )
        if early_stop == "val_recall_gal":
            # 4) Set up callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_recall_gal",
                    patience=es_patience,
                    mode="max",
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    mode="min",
                    factor=0.5,
                    patience=rlr_patience,
                    verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath="best_model.keras",
                    monitor="val_recall_gal",
                    mode="max",
                    save_best_only=True,
                    verbose=1
                )
            ]
        else:
            # 4) Set up callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=es_patience,
                    mode="min",
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    mode="min",
                    factor=0.5,
                    patience=rlr_patience,
                    verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath="best_model.keras",
                    monitor="val_loss",
                    mode="min",
                    save_best_only=True,
                    verbose=1
                )
            ]
        
        # Add gradual unfreezing callback if requested
        if gradual_unfreeze:
            callbacks.append(self._create_gradual_unfreeze_callback(num_unfreeze_layers))

        # 5) Fine-tune on training data, evaluate on separate validation data
        print("\n--- Starting Fine-tuning ---")
        self.history = self.model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),  # Use separate validation set
            shuffle=True,
            class_weight=class_weight,
            callbacks=callbacks
        )

        print("\n--- Fine-tuning complete ---")
        self.plot_training_history()

        return self.history

    def _create_gradual_unfreeze_callback(self, total_layers_to_unfreeze):
        """
        Create a callback that gradually unfreezes layers during training.
        
        Args:
            total_layers_to_unfreeze: Total number of layers to unfreeze
            
        Returns:
            A Keras callback
        """
        class GradualUnfreezeCallback(keras.callbacks.Callback):
            def __init__(self, base_model, total_layers):
                super().__init__()
                self.base_model = base_model
                self.total_layers = total_layers
                self.unfrozen_count = 0
                
            def on_epoch_end(self, epoch, logs=None):
                # Every 2 epochs, unfreeze next layer from the end
                if epoch > 0 and epoch % 2 == 0 and self.unfrozen_count < self.total_layers:
                    layer_to_unfreeze = -(self.unfrozen_count + 1)
                    if abs(layer_to_unfreeze) <= len(self.base_model.layers):
                        layer = self.base_model.layers[layer_to_unfreeze]
                        layer.trainable = True
                        self.unfrozen_count += 1
                        print(f"\nUnfreezing layer: {layer.name}")
                
        return GradualUnfreezeCallback(self.base_model, total_layers_to_unfreeze)
    
    def evaluate(self, X_test, y_test, batch_size=32, class_names=None):
        """
        Evaluate the model on test data and display detailed metrics.
        
        Args:
            X_test: Test data
            y_test: Test labels
            batch_size: Batch size for evaluation
            class_names: Optional list of class names for reporting
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("\n--- Model Evaluation ---")
        results = self.model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
        
        metrics = {}
        for name, val in zip(self.model.metrics_names, results):
            metrics[name] = val
            print(f"{name}: {val:.4f}")
            
        # Plot confusion matrix
        self.plot_confusion_matrix(X_test, y_test, class_names)
        
        return metrics
    
    def save_model(self, filepath="fine_tuned_model"):
        """
        Save the fine-tuned model to disk.
        
        Args:
            filepath: Path to save the model
                - If ends with .h5 or .keras: Saved as a single file
                - Otherwise: Saved as a directory containing the model
        """
        # Check if the path ends with a file extension
        is_file = filepath.endswith('.h5') or filepath.endswith('.keras')
        
        if not is_file:
            # Treat as directory path
            if not os.path.exists(filepath):
                os.makedirs(filepath)
        else:
            # For file path, ensure the directory exists
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath="fine_tuned_model"):
        """
        Load a previously saved model.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            The loaded model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model