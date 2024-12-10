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
- Achieved in ~15-17 epochs
- Final Test Accuracy: 99.4%+

## Requirements Met
✓ Less than 20k parameters (actual: 4,994)
✓ Uses Batch Normalization
✓ Uses Dropout
✓ Uses Global Average Pooling
✓ No Fully Connected Layers
✓ Achieves 99.4% accuracy
✓ Completes within 20 epochs
