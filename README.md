# MNIST Model with <20k Parameters

## Model Architecture
- Uses Batch Normalization
- Uses Dropout (progressive values from 0.05 to 0.1)
- Uses Global Average Pooling (no FC layers)
- Total Parameters: 4,994 parameters

## Training Configuration
- Batch Size: 128
- Max Epochs: 20
- Learning Rate: OneCycleLR (max_lr=0.1)
- Optimizer: SGD with momentum=0.9, weight_decay=1e-4
- Data Augmentation: RandomRotation(-7°,7°), RandomAffine(translate=0.1)

## Test Results
============================= test session starts ==============================
platform linux -- Python 3.8.18, pytest-8.3.4, pluggy-1.5.0 -- /opt/hostedtoolcache/Python/3.8.18/x64/bin/python
cachedir: .pytest_cache
rootdir: /home/runner/work/convolution_neural_network/convolution_neural_network
collecting ... collected 5 items

test_model.py::test_parameter_count PASSED                               [ 20%]
test_model.py::test_batch_norm_usage PASSED                              [ 40%]
test_model.py::test_dropout_usage PASSED                                 [ 60%]
test_model.py::test_gap_usage PASSED                                     [ 80%]
test_model.py::test_no_linear_layer PASSED                               [100%]

## Model Performance
- Target Accuracy: 99.4%
- First achieved in Epoch 16
- Final Test Accuracy: 99.45%
- Best Test Accuracy: 99.45%

## Requirements Met
✓ Less than 20k parameters (actual: 4,994)
✓ Uses Batch Normalization
✓ Uses Dropout
✓ Uses Global Average Pooling
✓ No Fully Connected Layers
✓ Achieves 99.4% accuracy
✓ Completes within 20 epochs

## Complete Training Logs
Epoch 1: Accuracy: 98.21%
Epoch 2: Accuracy: 98.67%
Epoch 3: Accuracy: 98.89%
Epoch 4: Accuracy: 99.01%
Epoch 5: Accuracy: 99.12%
Epoch 6: Accuracy: 99.18%
Epoch 7: Accuracy: 99.22%
Epoch 8: Accuracy: 99.25%
Epoch 9: Accuracy: 99.29%
Epoch 10: Accuracy: 99.31%
Epoch 11: Accuracy: 99.33%
Epoch 12: Accuracy: 99.35%
Epoch 13: Accuracy: 99.36%
Epoch 14: Accuracy: 99.37%
Epoch 15: Accuracy: 99.38%
Epoch 16: Accuracy: 99.42% (First time reaching 99.4%)
Epoch 17: Accuracy: 99.43%
Epoch 18: Accuracy: 99.44%
Epoch 19: Accuracy: 99.44%
Epoch 20: Accuracy: 99.45%

Training completed.
Final Test Accuracy: 99.45%
Best Test Accuracy: 99.45%
