import numpy as np
from nnfs.datasets import spiral_data
from my_N_N_library.data import Data_creation
from my_N_N_library import (
    Layer_Dense,
    Activation_ReLU,
    Activation_softmax_Loss_categoricalcrossentropy,
    optimizer_GD
)


def test_pipeline():
    print("--- Starting Neural Network Test ---")

    # 1. Generate synthetic dataset (100 samples per class, 3 classes, 2D points)
    X, y = spiral_data(100,3)
    print(f"Data shape: X = {X.shape}, y = {y.shape}")

    # 2. Instantiate network layers
    # Input layer: 2 inputs -> 3 hidden neurons
    dense1 = Layer_Dense(2, 64)
    activation1 = Activation_ReLU()

    # Output layer: 3 hidden inputs -> 3 output classes
    dense2 = Layer_Dense(64, 3)
    loss_activation = Activation_softmax_Loss_categoricalcrossentropy()

    optimizer=optimizer_GD()

    for epoch in range(10001):

        # 3. Perform Forward Pass
        # Layer 1
        dense1.forward(X)
        activation1.forward(dense1.output)

        # Layer 2 + Loss Calculation
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y)
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape)==2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)

        if not epoch % 100:
            print(f'epoch: {epoch},'+
                  f'accuracy:{accuracy:.3f}'+
                  f'loss:{loss:.3f}')


        #5. Backprogpagation implementation:
        loss_activation.backwards(loss_activation.output,y)
        dense2.backwards(loss_activation.dinputs)
        activation1.backwards(dense2.dinputs)
        dense1.backwards(activation1.dinputs)

        #6 parameter update
        optimizer.pre_update_parameters()
        optimizer.update_parameters(dense1)
        optimizer.update_parameters(dense2)
        optimizer.post_update_parameters()

test_pipeline()
