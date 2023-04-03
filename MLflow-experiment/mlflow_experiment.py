
"""
The code trains a convolutional neural network (CNN) on the MNIST dataset and uses 
MLflow to track the experiment. It reads in configuration parameters from an INI file
and logs them as parameters to the MLflow run. The model is trained using early
stopping and the validation loss is used as the metric for early stopping. The test
loss and accuracy are logged as metrics to the MLflow run, along with the number of
training, validation, and test samples. The CNN model and data preparation functions
are logged as artifacts to the MLflow run. To access the MLflow UI, run the following
command in the terminal:
```
mlflow server
```
MLflow logs can be accessed via: http://127.0.0.1:5000
"""

import tensorflow as tf
from cnn_model import build_cnnmodel
from data_prep import prepare_mnist_data
import mlflow
import configparser
from pyprojroot import here
import random

def main():
    # Read configuration parameters from the mlflow.cfg file
    config = configparser.ConfigParser()
    config.read(here("MLflow/mlflow.cfg"))
    random_seed = int(config["MLflowTutorial"]["random_seed"])
    random.seed(random_seed)
    learning_rate = float(config["MLflowTutorial"]["learning_rate"])
    early_stop_patience = int(config["MLflowTutorial"]["early_stop_patience"])
    optimizer = config["MLflowTutorial"]["optimizer"]
    num_batch_size = int(config["MLflowTutorial"]["num_batch_size"])
    num_epochs = int(config["MLflowTutorial"]["num_epochs"])
    experiment_name = config["MLflowTutorial"]["experiment_name"]

    # Prepare the MNIST data
    x_train, y_train, x_valid, y_valid, x_test, y_test = prepare_mnist_data()

    # Build the CNN model and compile it
    cnnmodel = build_cnnmodel()
    cnnmodel.compile(loss="categorical_crossentropy",
                      optimizer="RMSProp",
                      metrics=["accuracy"])
    
    # Set the name of the experiment
    mlflow.set_experiment(experiment_name)
    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_param("random_seed", random_seed)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("early_stop_patience", early_stop_patience)
        mlflow.log_param("optimizer", optimizer)
        mlflow.log_param("num_batch_size", num_batch_size)
        mlflow.log_param("num_epochs", num_epochs)
        # Define an EarlyStopping callback to stop training if the validation loss stops improving
        
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",  # metrics to monitor
            patience=early_stop_patience,  # how many epochs before stop
            verbose=1,
            mode="min",  # we need the minimum loss
            restore_best_weights=True,
        )
         
        # Train the model
        history = cnnmodel.fit(
            x_train,
            y_train,
            batch_size=num_batch_size,
            epochs=num_epochs,
            verbose=1,
            callbacks=[callback],
            validation_data=(x_valid, y_valid),
        )

        # Evaluate the model on the test data
        test_loss, test_acc = cnnmodel.evaluate(x_test, y_test, verbose=2)
        print("\nTest accuracy:", test_acc)
        mlflow.keras.log_model(cnnmodel, "cnnmodel")

        # Log additional information to MLflow
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_acc", test_acc)
        mlflow.log_metric("num_train_samples", len(x_train))
        mlflow.log_metric("num_val_samples", len(x_valid))
        mlflow.log_metric("num_test_samples", len(x_test))
        mlflow.log_artifact(here("MLflow/CNNModel.py"))
        mlflow.log_artifact(here("MLflow/DataPrep.py"))

        # End the MLflow run
        mlflow.end_run()

if __name__ == "__main__":
    main()
