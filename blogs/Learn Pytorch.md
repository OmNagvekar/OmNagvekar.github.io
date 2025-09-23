```python
import torch
from torch import nn
import matplotlib.pyplot as plt
```


```python
torch.__version__
```




    '2.1.2+cpu'




```python
# Create *known* parameters
weight = 0.7
bias = 0.3

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

X[:10], y[:10]
```




    (tensor([[0.0000],
             [0.0200],
             [0.0400],
             [0.0600],
             [0.0800],
             [0.1000],
             [0.1200],
             [0.1400],
             [0.1600],
             [0.1800]]),
     tensor([[0.3000],
             [0.3140],
             [0.3280],
             [0.3420],
             [0.3560],
             [0.3700],
             [0.3840],
             [0.3980],
             [0.4120],
             [0.4260]]))




```python
y.shape
```




    torch.Size([50, 1])




```python
# Create train/test split
train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing 
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test)
```




    (40, 40, 10, 10)




```python
def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))

  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

  if predictions is not None:
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  # Show the legend
  plt.legend(prop={"size": 14});
```


```python
plot_predictions()
```


    
![png](https://github.com/OmNagvekar/OmNagvekar.github.io/blob/26300d8babc4cca124cc4113de97850ec002af75/blogs/Learn%20Pytorch_files/Learn%20Pytorch_6_0.png)
    



```python
# Create a Linear Regression model class
class LinearRegressionModel(nn.Module): # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
    def __init__(self):
        super().__init__() 
        self.weights = nn.Parameter(torch.randn(1, # <- start with random weights (this will get adjusted as the model learns)
                                                dtype=torch.float), # <- PyTorch loves float32 by default
                                   requires_grad=True) # <- can we update this value with gradient descent?)

        self.bias = nn.Parameter(torch.randn(1, # <- start with random bias (this will get adjusted as the model learns)
                                            dtype=torch.float), # <- PyTorch loves float32 by default
                                requires_grad=True) # <- can we update this value with gradient descent?))

    # Forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data (e.g. training/testing features)
        return self.weights * x + self.bias # <- this is the linear regression formula (y = m*x + b)
```


```python
# Set manual seed since nn.Parameter are randomly initialzied
torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))
model_0 = LinearRegressionModel()

# Check the nn.Parameter(s) within the nn.Module subclass we created
list(model_0.parameters())
```




    [Parameter containing:
     tensor([0.3367], requires_grad=True),
     Parameter containing:
     tensor([0.1288], requires_grad=True)]




```python
# List named parameters 
model_0.state_dict()
```




    OrderedDict([('weights', tensor([0.3367])), ('bias', tensor([0.1288]))])




```python
# Make predictions with model
with torch.inference_mode(): 
    y_preds = model_0(X_test)

y_preds
```




    tensor([[0.3982],
            [0.4049],
            [0.4116],
            [0.4184],
            [0.4251],
            [0.4318],
            [0.4386],
            [0.4453],
            [0.4520],
            [0.4588]])




```python
plot_predictions(predictions=y_preds)
```


    
![png](Learn%20Pytorch_files/Learn%20Pytorch_11_0.png)
    



```python
# Create the loss function
loss_fn = nn.L1Loss() # MAE loss is same as L1Loss

# Create the optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), # parameters of target model to optimize
                            lr=0.01) # learning rate (how much the optimizer should change parameters at each step, higher=more (less stable), lower=less (might take a long time))
```


```python
from tqdm.auto import tqdm
torch.manual_seed(42)

# Set the number of epochs (how many times the model will pass over the training data)
epochs = 150

# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in tqdm(range(epochs)):
    ### Training

    # Put model in training mode (this is the default state of a model)
    model_0.train()

    # 1. Forward pass on train data using the forward() method inside 
    y_pred = model_0(X_train)
    # print(y_pred)

    # 2. Calculate the loss (how different are our models predictions to the ground truth)
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad of the optimizer
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Progress the optimizer
    optimizer.step()
    print(f"Epoch: {epoch}/{epochs}  Loss:{loss}")

    ### Testing

    # Put the model in evaluation mode
    model_0.eval()

    with torch.inference_mode():
      # 1. Forward pass on test data
      test_pred = model_0(X_test)

      # 2. Caculate loss on test data
      test_loss = loss_fn(test_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

      # Print out what's happening
      if epoch % 10 == 0:
        epoch_count.append(epoch)
        train_loss_values.append(loss.numpy())
        test_loss_values.append(test_loss.numpy())
        print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")
        print(model_0.state_dict())
```


      0%|          | 0/150 [00:00<?, ?it/s]


    Epoch: 0/150  Loss:0.31288138031959534
    Epoch: 0 | MAE Train Loss: 0.31288138031959534 | MAE Test Loss: 0.48106518387794495 
    OrderedDict([('weights', tensor([0.3406])), ('bias', tensor([0.1388]))])
    Epoch: 1/150  Loss:0.3013603389263153
    Epoch: 2/150  Loss:0.28983935713768005
    Epoch: 3/150  Loss:0.2783183455467224
    Epoch: 4/150  Loss:0.26679736375808716
    Epoch: 5/150  Loss:0.2552763521671295
    Epoch: 6/150  Loss:0.24375534057617188
    Epoch: 7/150  Loss:0.23223432898521423
    Epoch: 8/150  Loss:0.22071333229541779
    Epoch: 9/150  Loss:0.20919232070446014
    Epoch: 10/150  Loss:0.1976713240146637
    Epoch: 10 | MAE Train Loss: 0.1976713240146637 | MAE Test Loss: 0.3463551998138428 
    OrderedDict([('weights', tensor([0.3796])), ('bias', tensor([0.2388]))])
    Epoch: 11/150  Loss:0.18615034222602844
    Epoch: 12/150  Loss:0.1746293306350708
    Epoch: 13/150  Loss:0.16310831904411316
    Epoch: 14/150  Loss:0.1515873372554779
    Epoch: 15/150  Loss:0.14006635546684265
    Epoch: 16/150  Loss:0.1285453587770462
    Epoch: 17/150  Loss:0.11702437698841095
    Epoch: 18/150  Loss:0.1060912236571312
    Epoch: 19/150  Loss:0.09681284427642822
    Epoch: 20/150  Loss:0.08908725529909134
    Epoch: 20 | MAE Train Loss: 0.08908725529909134 | MAE Test Loss: 0.21729660034179688 
    OrderedDict([('weights', tensor([0.4184])), ('bias', tensor([0.3333]))])
    Epoch: 21/150  Loss:0.08227583020925522
    Epoch: 22/150  Loss:0.07638873159885406
    Epoch: 23/150  Loss:0.07160007208585739
    Epoch: 24/150  Loss:0.06747635453939438
    Epoch: 25/150  Loss:0.06395438313484192
    Epoch: 26/150  Loss:0.06097004935145378
    Epoch: 27/150  Loss:0.05845819041132927
    Epoch: 28/150  Loss:0.05635259300470352
    Epoch: 29/150  Loss:0.0545857772231102
    Epoch: 30/150  Loss:0.053148526698350906
    Epoch: 30 | MAE Train Loss: 0.053148526698350906 | MAE Test Loss: 0.14464017748832703 
    OrderedDict([('weights', tensor([0.4512])), ('bias', tensor([0.3768]))])
    Epoch: 31/150  Loss:0.05181945487856865
    Epoch: 32/150  Loss:0.05069301277399063
    Epoch: 33/150  Loss:0.0498228520154953
    Epoch: 34/150  Loss:0.04895269125699997
    Epoch: 35/150  Loss:0.04819351062178612
    Epoch: 36/150  Loss:0.047531817108392715
    Epoch: 37/150  Loss:0.04692792519927025
    Epoch: 38/150  Loss:0.04642331600189209
    Epoch: 39/150  Loss:0.04591871052980423
    Epoch: 40/150  Loss:0.04543796554207802
    Epoch: 40 | MAE Train Loss: 0.04543796554207802 | MAE Test Loss: 0.11360953003168106 
    OrderedDict([('weights', tensor([0.4748])), ('bias', tensor([0.3868]))])
    Epoch: 41/150  Loss:0.04503796249628067
    Epoch: 42/150  Loss:0.04463795945048332
    Epoch: 43/150  Loss:0.04423796385526657
    Epoch: 44/150  Loss:0.04383796453475952
    Epoch: 45/150  Loss:0.04343796148896217
    Epoch: 46/150  Loss:0.043074630200862885
    Epoch: 47/150  Loss:0.04272563382983208
    Epoch: 48/150  Loss:0.04237663000822067
    Epoch: 49/150  Loss:0.04202762991189957
    Epoch: 50/150  Loss:0.04167863354086876
    Epoch: 50 | MAE Train Loss: 0.04167863354086876 | MAE Test Loss: 0.09919948130846024 
    OrderedDict([('weights', tensor([0.4938])), ('bias', tensor([0.3843]))])
    Epoch: 51/150  Loss:0.04132963344454765
    Epoch: 52/150  Loss:0.04098063334822655
    Epoch: 53/150  Loss:0.04063162952661514
    Epoch: 54/150  Loss:0.040282636880874634
    Epoch: 55/150  Loss:0.039933640509843826
    Epoch: 56/150  Loss:0.03958464413881302
    Epoch: 57/150  Loss:0.03923564404249191
    Epoch: 58/150  Loss:0.03888664394617081
    Epoch: 59/150  Loss:0.0385376438498497
    Epoch: 60/150  Loss:0.03818932920694351
    Epoch: 60 | MAE Train Loss: 0.03818932920694351 | MAE Test Loss: 0.08886633068323135 
    OrderedDict([('weights', tensor([0.5116])), ('bias', tensor([0.3788]))])
    Epoch: 61/150  Loss:0.03785243630409241
    Epoch: 62/150  Loss:0.0375034399330616
    Epoch: 63/150  Loss:0.037164121866226196
    Epoch: 64/150  Loss:0.03681822493672371
    Epoch: 65/150  Loss:0.03647511452436447
    Epoch: 66/150  Loss:0.03613303601741791
    Epoch: 67/150  Loss:0.03578609973192215
    Epoch: 68/150  Loss:0.03544783592224121
    Epoch: 69/150  Loss:0.035098835825920105
    Epoch: 70/150  Loss:0.03476089984178543
    Epoch: 70 | MAE Train Loss: 0.03476089984178543 | MAE Test Loss: 0.0805937647819519 
    OrderedDict([('weights', tensor([0.5288])), ('bias', tensor([0.3718]))])
    Epoch: 71/150  Loss:0.03441363573074341
    Epoch: 72/150  Loss:0.03407188132405281
    Epoch: 73/150  Loss:0.03372843936085701
    Epoch: 74/150  Loss:0.03338287025690079
    Epoch: 75/150  Loss:0.033043231815099716
    Epoch: 76/150  Loss:0.03269423171877861
    Epoch: 77/150  Loss:0.032357655465602875
    Epoch: 78/150  Loss:0.03200903534889221
    Epoch: 79/150  Loss:0.03166864812374115
    Epoch: 80/150  Loss:0.03132382780313492
    Epoch: 80 | MAE Train Loss: 0.03132382780313492 | MAE Test Loss: 0.07232122868299484 
    OrderedDict([('weights', tensor([0.5459])), ('bias', tensor([0.3648]))])
    Epoch: 81/150  Loss:0.030979642644524574
    Epoch: 82/150  Loss:0.030638623982667923
    Epoch: 83/150  Loss:0.0302906334400177
    Epoch: 84/150  Loss:0.029953425750136375
    Epoch: 85/150  Loss:0.02960442565381527
    Epoch: 86/150  Loss:0.029265418648719788
    Epoch: 87/150  Loss:0.028919223695993423
    Epoch: 88/150  Loss:0.028576409444212914
    Epoch: 89/150  Loss:0.028234025463461876
    Epoch: 90/150  Loss:0.02788739837706089
    Epoch: 90 | MAE Train Loss: 0.02788739837706089 | MAE Test Loss: 0.06473556160926819 
    OrderedDict([('weights', tensor([0.5629])), ('bias', tensor([0.3573]))])
    Epoch: 91/150  Loss:0.02754882536828518
    Epoch: 92/150  Loss:0.027199819684028625
    Epoch: 93/150  Loss:0.026862185448408127
    Epoch: 94/150  Loss:0.02651461586356163
    Epoch: 95/150  Loss:0.026173178106546402
    Epoch: 96/150  Loss:0.025829419493675232
    Epoch: 97/150  Loss:0.02548416517674923
    Epoch: 98/150  Loss:0.025144213810563087
    Epoch: 99/150  Loss:0.02479521557688713
    Epoch: 100/150  Loss:0.024458957836031914
    Epoch: 100 | MAE Train Loss: 0.024458957836031914 | MAE Test Loss: 0.05646304413676262 
    OrderedDict([('weights', tensor([0.5800])), ('bias', tensor([0.3503]))])
    Epoch: 101/150  Loss:0.024110013619065285
    Epoch: 102/150  Loss:0.02376994863152504
    Epoch: 103/150  Loss:0.02342480979859829
    Epoch: 104/150  Loss:0.023080935701727867
    Epoch: 105/150  Loss:0.022739607840776443
    Epoch: 106/150  Loss:0.022391926497220993
    Epoch: 107/150  Loss:0.022054409608244896
    Epoch: 108/150  Loss:0.02170540764927864
    Epoch: 109/150  Loss:0.021366719156503677
    Epoch: 110/150  Loss:0.021020207554101944
    Epoch: 110 | MAE Train Loss: 0.021020207554101944 | MAE Test Loss: 0.04819049686193466 
    OrderedDict([('weights', tensor([0.5972])), ('bias', tensor([0.3433]))])
    Epoch: 111/150  Loss:0.020677709951996803
    Epoch: 112/150  Loss:0.02033500373363495
    Epoch: 113/150  Loss:0.01998869702219963
    Epoch: 114/150  Loss:0.019649803638458252
    Epoch: 115/150  Loss:0.019300809130072594
    Epoch: 116/150  Loss:0.018963487818837166
    Epoch: 117/150  Loss:0.01861560344696045
    Epoch: 118/150  Loss:0.018274478614330292
    Epoch: 119/150  Loss:0.017930403351783752
    Epoch: 120/150  Loss:0.01758546568453312
    Epoch: 120 | MAE Train Loss: 0.01758546568453312 | MAE Test Loss: 0.04060482233762741 
    OrderedDict([('weights', tensor([0.6141])), ('bias', tensor([0.3358]))])
    Epoch: 121/150  Loss:0.017245199531316757
    Epoch: 122/150  Loss:0.016896454617381096
    Epoch: 123/150  Loss:0.01656000316143036
    Epoch: 124/150  Loss:0.016210997477173805
    Epoch: 125/150  Loss:0.01587124727666378
    Epoch: 126/150  Loss:0.015525798313319683
    Epoch: 127/150  Loss:0.015182236209511757
    Epoch: 128/150  Loss:0.014840595424175262
    Epoch: 129/150  Loss:0.01449323259294033
    Epoch: 130/150  Loss:0.014155393466353416
    Epoch: 130 | MAE Train Loss: 0.014155393466353416 | MAE Test Loss: 0.03233227878808975 
    OrderedDict([('weights', tensor([0.6313])), ('bias', tensor([0.3288]))])
    Epoch: 131/150  Loss:0.013806397095322609
    Epoch: 132/150  Loss:0.013468016870319843
    Epoch: 133/150  Loss:0.013121193274855614
    Epoch: 134/150  Loss:0.01277900766581297
    Epoch: 135/150  Loss:0.012435992248356342
    Epoch: 136/150  Loss:0.01208999752998352
    Epoch: 137/150  Loss:0.011750795878469944
    Epoch: 138/150  Loss:0.011401787400245667
    Epoch: 139/150  Loss:0.011064787395298481
    Epoch: 140/150  Loss:0.010716589167714119
    Epoch: 140 | MAE Train Loss: 0.010716589167714119 | MAE Test Loss: 0.024059748277068138 
    OrderedDict([('weights', tensor([0.6485])), ('bias', tensor([0.3218]))])
    Epoch: 141/150  Loss:0.010375778190791607
    Epoch: 142/150  Loss:0.010031387209892273
    Epoch: 143/150  Loss:0.009686763398349285
    Epoch: 144/150  Loss:0.009346187114715576
    Epoch: 145/150  Loss:0.008997755125164986
    Epoch: 146/150  Loss:0.008660981431603432
    Epoch: 147/150  Loss:0.008311985060572624
    Epoch: 148/150  Loss:0.007972544990479946
    Epoch: 149/150  Loss:0.007626785431057215
    


```python
plot_predictions(predictions=test_pred)
```


    
![png](Learn%20Pytorch_files/Learn%20Pytorch_14_0.png)
    



```python
# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend();
```


    
![png](Learn%20Pytorch_files/Learn%20Pytorch_15_0.png)
    



```python
# Find our model's learned parameters
print("The model learned the following values for weights and bias:")
print(model_0.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")
```

    The model learned the following values for weights and bias:
    OrderedDict([('weights', tensor([0.6638])), ('bias', tensor([0.3153]))])
    
    And the original values for weights and bias are:
    weights: 0.7, bias: 0.3
    


```python
from pathlib import Path

# 1. Create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path 
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH) 
```

    Saving model to: models\01_pytorch_workflow_model_0.pth
    


```python
torch.save(model_0,"models/01_pytorch_workflow_model_0.pkl")
c=torch.load("models/01_pytorch_workflow_model_0.pkl")
c.eval()
with torch.inference_mode():
    print(c(X_test).cpu())
```

    tensor([[0.8464],
            [0.8596],
            [0.8729],
            [0.8862],
            [0.8995],
            [0.9127],
            [0.9260],
            [0.9393],
            [0.9526],
            [0.9659]])
    


```python
# Instantiate a new instance of our model (this will be instantiated with random weights)
loaded_model_0 = LinearRegressionModel()

# Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
```




    <All keys matched successfully>




```python
# 1. Put the loaded model into evaluation mode
loaded_model_0.eval()

# 2. Use the inference mode context manager to make predictions
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test) # perform a forward pass on the test data with the loaded model
```


```python
# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

    Using device: cpu
    


```python
# Check model device
next(model_0.parameters()).device
```




    device(type='cpu')




```python
# Set model to GPU if it's availalble, otherwise it'll default to CPU
model_1.to(device) # the device variable was set above to be "cuda" if available or "cpu" if not
next(model_1.parameters()).device
```




    device(type='cpu')




```python
X_train.shape
```


```python
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1,out_features=1)

    def forward(self,x:torch.float)->torch.Tensor:
        return self.linear_layer(x)
```


```python
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
```


```python
model_1,model_1.state_dict()
```




    (LinearRegressionModelV2(
       (linear_layer): Linear(in_features=1, out_features=1, bias=True)
     ),
     OrderedDict([('linear_layer.weight', tensor([[0.7645]])),
                  ('linear_layer.bias', tensor([0.8300]))]))




```python
epochs_2 =150
loss_fn_2 = nn.L1Loss()
optimizer_2 = torch.optim.Adam(params=model_1.parameters(),lr=0.01)
loss_fn_2,optimizer_2
```




    (L1Loss(),
     Adam (
     Parameter Group 0
         amsgrad: False
         betas: (0.9, 0.999)
         capturable: False
         differentiable: False
         eps: 1e-08
         foreach: None
         fused: None
         lr: 0.01
         maximize: False
         weight_decay: 0
     ))




```python
torch.manual_seed(42)
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs_2):
    model_1.train()
    y_pred = model_1(X_train)
    loss = loss_fn_2(y_pred,y_train)
    optimizer_2.zero_grad()
    loss.backward()
    optimizer_2.step()
    

    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)
        test_loss = loss_fn_2(test_pred,y_test)
        accuracy =  1-((torch.sum(torch.pow((y_test-test_pred),2)))/(torch.sum(torch.pow((y_test-y_test.mean()),2))))
        epoch_count.append(epoch)
        train_loss_values.append(loss.numpy())
        test_loss_values.append(test_loss.numpy())
    print(f"epoch:{epoch+1}/{epochs_2} | Loss: {loss} | Test loss:{test_loss} | test accuracy: {accuracy*100}")
```

    epoch:1/150 | Loss: 0.5551779866218567 | Test loss:0.568547248840332 | test accuracy: -19891.08203125
    epoch:2/150 | Loss: 0.5412780046463013 | Test loss:0.5496472120285034 | test accuracy: -18583.8984375
    epoch:3/150 | Loss: 0.5273779630661011 | Test loss:0.5307471752166748 | test accuracy: -17320.94140625
    epoch:4/150 | Loss: 0.5134779810905457 | Test loss:0.5118472576141357 | test accuracy: -16102.205078125
    epoch:5/150 | Loss: 0.499578058719635 | Test loss:0.4929472804069519 | test accuracy: -14927.6943359375
    epoch:6/150 | Loss: 0.4856780171394348 | Test loss:0.47404733300209045 | test accuracy: -13797.4033203125
    epoch:7/150 | Loss: 0.4717780649662018 | Test loss:0.45514732599258423 | test accuracy: -12711.3310546875
    epoch:8/150 | Loss: 0.4578780233860016 | Test loss:0.4362473487854004 | test accuracy: -11669.4853515625
    epoch:9/150 | Loss: 0.4439780116081238 | Test loss:0.41734737157821655 | test accuracy: -10671.8603515625
    epoch:10/150 | Loss: 0.43007802963256836 | Test loss:0.3984473943710327 | test accuracy: -9718.458984375
    epoch:11/150 | Loss: 0.4161780774593353 | Test loss:0.3795473873615265 | test accuracy: -8809.28125
    epoch:12/150 | Loss: 0.40227803587913513 | Test loss:0.36064741015434265 | test accuracy: -7944.32470703125
    epoch:13/150 | Loss: 0.3883780837059021 | Test loss:0.3417474627494812 | test accuracy: -7123.591796875
    epoch:14/150 | Loss: 0.37447813153266907 | Test loss:0.3228474259376526 | test accuracy: -6347.080078125
    epoch:15/150 | Loss: 0.36057811975479126 | Test loss:0.30394744873046875 | test accuracy: -5614.7919921875
    epoch:16/150 | Loss: 0.3466781675815582 | Test loss:0.2850474715232849 | test accuracy: -4926.72607421875
    epoch:17/150 | Loss: 0.33277812600135803 | Test loss:0.26614755392074585 | test accuracy: -4282.8828125
    epoch:18/150 | Loss: 0.318878173828125 | Test loss:0.24724750220775604 | test accuracy: -3683.26123046875
    epoch:19/150 | Loss: 0.3049781918525696 | Test loss:0.2283475399017334 | test accuracy: -3127.863037109375
    epoch:20/150 | Loss: 0.2910781800746918 | Test loss:0.20944757759571075 | test accuracy: -2616.688232421875
    epoch:21/150 | Loss: 0.27717819809913635 | Test loss:0.19054758548736572 | test accuracy: -2149.734375
    epoch:22/150 | Loss: 0.26327821612358093 | Test loss:0.1716475933790207 | test accuracy: -1727.00439453125
    epoch:23/150 | Loss: 0.2493782341480255 | Test loss:0.15274760127067566 | test accuracy: -1348.4964599609375
    epoch:24/150 | Loss: 0.2354782521724701 | Test loss:0.13384762406349182 | test accuracy: -1014.2106323242188
    epoch:25/150 | Loss: 0.22157824039459229 | Test loss:0.11494765430688858 | test accuracy: -724.1478271484375
    epoch:26/150 | Loss: 0.20767827332019806 | Test loss:0.09604767709970474 | test accuracy: -478.3076171875
    epoch:27/150 | Loss: 0.19377827644348145 | Test loss:0.07714769244194031 | test accuracy: -276.6898193359375
    epoch:28/150 | Loss: 0.17987829446792603 | Test loss:0.058247704058885574 | test accuracy: -119.29457092285156
    epoch:29/150 | Loss: 0.1659782975912094 | Test loss:0.03934771567583084 | test accuracy: -6.121861934661865
    epoch:30/150 | Loss: 0.152078315615654 | Test loss:0.02059648558497429 | test accuracy: 62.82817077636719
    epoch:31/150 | Loss: 0.13817831873893738 | Test loss:0.012273055501282215 | test accuracy: 87.55570220947266
    epoch:32/150 | Loss: 0.12427835166454315 | Test loss:0.01858610473573208 | test accuracy: 68.0606460571289
    epoch:33/150 | Loss: 0.11037836223840714 | Test loss:0.036252208054065704 | test accuracy: 4.343074798583984
    epoch:34/150 | Loss: 0.09691805392503738 | Test loss:0.05494359880685806 | test accuracy: -102.15990447998047
    epoch:35/150 | Loss: 0.08612346649169922 | Test loss:0.07312221825122833 | test accuracy: -247.2177734375
    epoch:36/150 | Loss: 0.07801749557256699 | Test loss:0.09050118178129196 | test accuracy: -424.1266174316406
    epoch:37/150 | Loss: 0.07223037630319595 | Test loss:0.10692725330591202 | test accuracy: -625.6824340820312
    epoch:38/150 | Loss: 0.06840451061725616 | Test loss:0.12214148044586182 | test accuracy: -842.1331176757812
    epoch:39/150 | Loss: 0.06612894684076309 | Test loss:0.13602332770824432 | test accuracy: -1064.591552734375
    epoch:40/150 | Loss: 0.06501899659633636 | Test loss:0.14846496284008026 | test accuracy: -1284.1986083984375
    epoch:41/150 | Loss: 0.06483234465122223 | Test loss:0.15948258340358734 | test accuracy: -1494.6209716796875
    epoch:42/150 | Loss: 0.06517203897237778 | Test loss:0.1690920889377594 | test accuracy: -1690.366943359375
    epoch:43/150 | Loss: 0.06593725085258484 | Test loss:0.1772030144929886 | test accuracy: -1864.423583984375
    epoch:44/150 | Loss: 0.06686188280582428 | Test loss:0.18384429812431335 | test accuracy: -2012.9417724609375
    epoch:45/150 | Loss: 0.06779052317142487 | Test loss:0.1890452355146408 | test accuracy: -2132.9853515625
    epoch:46/150 | Loss: 0.06866536289453506 | Test loss:0.19293558597564697 | test accuracy: -2224.8818359375
    epoch:47/150 | Loss: 0.06935916841030121 | Test loss:0.19553284347057343 | test accuracy: -2287.16650390625
    epoch:48/150 | Loss: 0.06980307400226593 | Test loss:0.19695408642292023 | test accuracy: -2321.454833984375
    epoch:49/150 | Loss: 0.06999608129262924 | Test loss:0.19720736145973206 | test accuracy: -2327.32763671875
    epoch:50/150 | Loss: 0.06992020457983017 | Test loss:0.19639885425567627 | test accuracy: -2307.264892578125
    epoch:51/150 | Loss: 0.06953977048397064 | Test loss:0.19462481141090393 | test accuracy: -2263.91064453125
    epoch:52/150 | Loss: 0.06888256222009659 | Test loss:0.19197258353233337 | test accuracy: -2199.984619140625
    epoch:53/150 | Loss: 0.06797375530004501 | Test loss:0.18852141499519348 | test accuracy: -2118.204833984375
    epoch:54/150 | Loss: 0.0668361634016037 | Test loss:0.18434305489063263 | test accuracy: -2021.2412109375
    epoch:55/150 | Loss: 0.0654904916882515 | Test loss:0.17950275540351868 | test accuracy: -1911.6749267578125
    epoch:56/150 | Loss: 0.06398600339889526 | Test loss:0.17415912449359894 | test accuracy: -1794.1220703125
    epoch:57/150 | Loss: 0.06241095811128616 | Test loss:0.16835620999336243 | test accuracy: -1670.503173828125
    epoch:58/150 | Loss: 0.06071857735514641 | Test loss:0.16213393211364746 | test accuracy: -1542.6134033203125
    epoch:59/150 | Loss: 0.05901934579014778 | Test loss:0.1556323915719986 | test accuracy: -1414.125
    epoch:60/150 | Loss: 0.057294946163892746 | Test loss:0.14887471497058868 | test accuracy: -1286.140625
    epoch:61/150 | Loss: 0.0556403286755085 | Test loss:0.14198976755142212 | test accuracy: -1161.5736083984375
    epoch:62/150 | Loss: 0.05401618406176567 | Test loss:0.13509811460971832 | test accuracy: -1042.769775390625
    epoch:63/150 | Loss: 0.05253006890416145 | Test loss:0.12819793820381165 | test accuracy: -929.7176513671875
    epoch:64/150 | Loss: 0.05114681273698807 | Test loss:0.1214035302400589 | test accuracy: -824.1609497070312
    epoch:65/150 | Loss: 0.04986557364463806 | Test loss:0.11482276767492294 | test accuracy: -727.3696899414062
    epoch:66/150 | Loss: 0.04873280972242355 | Test loss:0.10843416303396225 | test accuracy: -638.5322265625
    epoch:67/150 | Loss: 0.04773203283548355 | Test loss:0.10234272480010986 | test accuracy: -558.526611328125
    epoch:68/150 | Loss: 0.0468226857483387 | Test loss:0.09664823114871979 | test accuracy: -487.87750244140625
    epoch:69/150 | Loss: 0.046005960553884506 | Test loss:0.09131307899951935 | test accuracy: -425.3182067871094
    epoch:70/150 | Loss: 0.04528895765542984 | Test loss:0.08643670380115509 | test accuracy: -371.20367431640625
    epoch:71/150 | Loss: 0.0445832721889019 | Test loss:0.08211381733417511 | test accuracy: -325.66815185546875
    epoch:72/150 | Loss: 0.04392357915639877 | Test loss:0.07829321175813675 | test accuracy: -287.3240966796875
    epoch:73/150 | Loss: 0.04321376234292984 | Test loss:0.07507087290287018 | test accuracy: -256.3530578613281
    epoch:74/150 | Loss: 0.04250415042042732 | Test loss:0.07239184528589249 | test accuracy: -231.5433807373047
    epoch:75/150 | Loss: 0.04171803221106529 | Test loss:0.0702061578631401 | test accuracy: -211.91761779785156
    epoch:76/150 | Loss: 0.04086191579699516 | Test loss:0.06846846640110016 | test accuracy: -196.68849182128906
    epoch:77/150 | Loss: 0.039941709488630295 | Test loss:0.0671376883983612 | test accuracy: -185.22300720214844
    epoch:78/150 | Loss: 0.03896277770400047 | Test loss:0.06617651879787445 | test accuracy: -177.01266479492188
    epoch:79/150 | Loss: 0.03792997822165489 | Test loss:0.06555113196372986 | test accuracy: -171.6492919921875
    epoch:80/150 | Loss: 0.03684772923588753 | Test loss:0.06523077189922333 | test accuracy: -168.8048553466797
    epoch:81/150 | Loss: 0.03573416918516159 | Test loss:0.06503848731517792 | test accuracy: -167.0106201171875
    epoch:82/150 | Loss: 0.03464270755648613 | Test loss:0.06496253609657288 | test accuracy: -166.16256713867188
    epoch:83/150 | Loss: 0.03353287652134895 | Test loss:0.06499239057302475 | test accuracy: -166.1721954345703
    epoch:84/150 | Loss: 0.03246375173330307 | Test loss:0.06496988236904144 | test accuracy: -165.76266479492188
    epoch:85/150 | Loss: 0.03140534833073616 | Test loss:0.06489984691143036 | test accuracy: -164.97369384765625
    epoch:86/150 | Loss: 0.030396005138754845 | Test loss:0.06463975459337234 | test accuracy: -162.66148376464844
    epoch:87/150 | Loss: 0.02940061129629612 | Test loss:0.06420701742172241 | test accuracy: -158.98385620117188
    epoch:88/150 | Loss: 0.028445804491639137 | Test loss:0.06347231566905975 | test accuracy: -152.9490509033203
    epoch:89/150 | Loss: 0.02749496139585972 | Test loss:0.062463320791721344 | test accuracy: -144.85498046875
    epoch:90/150 | Loss: 0.026546627283096313 | Test loss:0.06106187775731087 | test accuracy: -133.90896606445312
    epoch:91/150 | Loss: 0.025599658489227295 | Test loss:0.05930420011281967 | test accuracy: -120.58291625976562
    epoch:92/150 | Loss: 0.02462359145283699 | Test loss:0.05722300335764885 | test accuracy: -105.3426742553711
    epoch:93/150 | Loss: 0.02362089231610298 | Test loss:0.0548480749130249 | test accuracy: -88.64208221435547
    epoch:94/150 | Loss: 0.022593818604946136 | Test loss:0.052206240594387054 | test accuracy: -70.91490173339844
    epoch:95/150 | Loss: 0.021544383838772774 | Test loss:0.04932183027267456 | test accuracy: -52.57273864746094
    epoch:96/150 | Loss: 0.02047443389892578 | Test loss:0.04621688649058342 | test accuracy: -34.00265121459961
    epoch:97/150 | Loss: 0.01938561163842678 | Test loss:0.04291144013404846 | test accuracy: -15.566312789916992
    epoch:98/150 | Loss: 0.018279436975717545 | Test loss:0.03942357376217842 | test accuracy: 2.4010062217712402
    epoch:99/150 | Loss: 0.017188403755426407 | Test loss:0.035918522626161575 | test accuracy: 18.927305221557617
    epoch:100/150 | Loss: 0.016099631786346436 | Test loss:0.032397132366895676 | test accuracy: 33.98664474487305
    epoch:101/150 | Loss: 0.015041215345263481 | Test loss:0.02901391312479973 | test accuracy: 47.002254486083984
    epoch:102/150 | Loss: 0.013992590829730034 | Test loss:0.025913000106811523 | test accuracy: 57.68535232543945
    epoch:103/150 | Loss: 0.012963414192199707 | Test loss:0.023067444562911987 | test accuracy: 66.43890380859375
    epoch:104/150 | Loss: 0.011924061924219131 | Test loss:0.02061549387872219 | test accuracy: 73.18040466308594
    epoch:105/150 | Loss: 0.010867838747799397 | Test loss:0.018519829958677292 | test accuracy: 78.35514831542969
    epoch:106/150 | Loss: 0.009776226244866848 | Test loss:0.016746724024415016 | test accuracy: 82.31150817871094
    epoch:107/150 | Loss: 0.008652105927467346 | Test loss:0.01526561938226223 | test accuracy: 85.32064819335938
    epoch:108/150 | Loss: 0.007519139908254147 | Test loss:0.013883257284760475 | test accuracy: 87.8793716430664
    epoch:109/150 | Loss: 0.00640082499012351 | Test loss:0.01242750883102417 | test accuracy: 90.30592346191406
    epoch:110/150 | Loss: 0.005318976938724518 | Test loss:0.010745054110884666 | test accuracy: 92.76622772216797
    epoch:111/150 | Loss: 0.004283720161765814 | Test loss:0.008546513505280018 | test accuracy: 95.4299087524414
    epoch:112/150 | Loss: 0.0032374695874750614 | Test loss:0.0055831135250627995 | test accuracy: 98.05023956298828
    epoch:113/150 | Loss: 0.002096372190862894 | Test loss:0.0019279122352600098 | test accuracy: 99.76631164550781
    epoch:114/150 | Loss: 0.0008829422295093536 | Test loss:0.0015798270469531417 | test accuracy: 99.84508514404297
    epoch:115/150 | Loss: 0.000744406133890152 | Test loss:0.001630634069442749 | test accuracy: 99.82930755615234
    epoch:116/150 | Loss: 0.00139881600625813 | Test loss:0.002719396259635687 | test accuracy: 99.52538299560547
    epoch:117/150 | Loss: 0.0023216926492750645 | Test loss:0.004744356963783503 | test accuracy: 98.57508850097656
    epoch:118/150 | Loss: 0.0027913928497582674 | Test loss:0.006877523846924305 | test accuracy: 97.0251693725586
    epoch:119/150 | Loss: 0.003187078982591629 | Test loss:0.00858951173722744 | test accuracy: 95.37347412109375
    epoch:120/150 | Loss: 0.0035390376579016447 | Test loss:0.009591621346771717 | test accuracy: 94.23802185058594
    epoch:121/150 | Loss: 0.0037691467441618443 | Test loss:0.009793078526854515 | test accuracy: 93.99445343017578
    epoch:122/150 | Loss: 0.003824451472610235 | Test loss:0.009115475229918957 | test accuracy: 94.79161834716797
    epoch:123/150 | Loss: 0.003696960862725973 | Test loss:0.007800537161529064 | test accuracy: 96.17707061767578
    epoch:124/150 | Loss: 0.00346815655939281 | Test loss:0.006233662366867065 | test accuracy: 97.54924011230469
    epoch:125/150 | Loss: 0.003215635661035776 | Test loss:0.0049529909156262875 | test accuracy: 98.44705963134766
    epoch:126/150 | Loss: 0.0029114573262631893 | Test loss:0.0041100322268903255 | test accuracy: 98.92996215820312
    epoch:127/150 | Loss: 0.0024693564046174288 | Test loss:0.0038458644412457943 | test accuracy: 99.06840515136719
    epoch:128/150 | Loss: 0.0018877685070037842 | Test loss:0.003565549850463867 | test accuracy: 99.20437622070312
    epoch:129/150 | Loss: 0.0013629526365548372 | Test loss:0.0024411380290985107 | test accuracy: 99.62848663330078
    epoch:130/150 | Loss: 0.0008563116425648332 | Test loss:0.0003226459084544331 | test accuracy: 99.99356079101562
    epoch:131/150 | Loss: 0.0002783544478006661 | Test loss:0.0004135310591664165 | test accuracy: 99.98857879638672
    epoch:132/150 | Loss: 0.0014299176400527358 | Test loss:0.002122140023857355 | test accuracy: 99.71571350097656
    epoch:133/150 | Loss: 0.001132487552240491 | Test loss:0.004546552896499634 | test accuracy: 98.70879364013672
    epoch:134/150 | Loss: 0.0016391829121857882 | Test loss:0.005406218580901623 | test accuracy: 98.17520141601562
    epoch:135/150 | Loss: 0.0019282742869108915 | Test loss:0.004856562707573175 | test accuracy: 98.52379608154297
    epoch:136/150 | Loss: 0.001861725002527237 | Test loss:0.0034913718700408936 | test accuracy: 99.23152160644531
    epoch:137/150 | Loss: 0.0017744272481650114 | Test loss:0.002221065806224942 | test accuracy: 99.6841049194336
    epoch:138/150 | Loss: 0.0018022290896624327 | Test loss:0.0021551132667809725 | test accuracy: 99.70442962646484
    epoch:139/150 | Loss: 0.0014897726941853762 | Test loss:0.002788710640743375 | test accuracy: 99.5127944946289
    epoch:140/150 | Loss: 0.001107488526031375 | Test loss:0.002644914435222745 | test accuracy: 99.56417083740234
    epoch:141/150 | Loss: 0.0009311929461546242 | Test loss:0.0006192505243234336 | test accuracy: 99.97596740722656
    epoch:142/150 | Loss: 0.0002487949968781322 | Test loss:0.0019224106799811125 | test accuracy: 99.77062225341797
    epoch:143/150 | Loss: 0.0009169928962364793 | Test loss:0.0009638607734814286 | test accuracy: 99.94023895263672
    epoch:144/150 | Loss: 0.000863347202539444 | Test loss:0.001387566328048706 | test accuracy: 99.87606048583984
    epoch:145/150 | Loss: 0.0012609653640538454 | Test loss:0.003056472633033991 | test accuracy: 99.4134750366211
    epoch:146/150 | Loss: 0.001321158604696393 | Test loss:0.004175060894340277 | test accuracy: 98.9107437133789
    epoch:147/150 | Loss: 0.001521474914625287 | Test loss:0.0039907037280499935 | test accuracy: 99.00457000732422
    epoch:148/150 | Loss: 0.0014644041657447815 | Test loss:0.002632051706314087 | test accuracy: 99.56437683105469
    epoch:149/150 | Loss: 0.0012117341393604875 | Test loss:0.001192587660625577 | test accuracy: 99.90834045410156
    epoch:150/150 | Loss: 0.0011099189287051558 | Test loss:0.0011917591327801347 | test accuracy: 99.91026306152344
    


```python
plot_predictions(predictions=test_pred)
```


    
![png](Learn%20Pytorch_files/Learn%20Pytorch_30_0.png)
    



```python
# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend();
```


    
![png](Learn%20Pytorch_files/Learn%20Pytorch_31_0.png)
    



```python
model_1.state_dict()
```




    OrderedDict([('linear_layer.weight', tensor([[0.7030]])),
                 ('linear_layer.bias', tensor([0.2985]))])




```python
from pathlib import Path

# 1. Create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path 
MODEL_NAME = "01_pytorch_workflow_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH) 
torch.save(model_1,"models/01_pytorch_workflow_model_1.pkl")
```

    Saving model to: models\01_pytorch_workflow_model_1.pth
    


```python
c=torch.load("models/01_pytorch_workflow_model_1.pkl")
d= LinearRegressionModelV2()
d.load_state_dict(torch.load("models/01_pytorch_workflow_model_1.pth"))
c.eval()
d.eval()
with torch.inference_mode():
    print("pickle file:",c(X_test))
    print("weights_file:",d(X_test))
```

    pickle file: tensor([[0.8609],
            [0.8750],
            [0.8890],
            [0.9031],
            [0.9172],
            [0.9312],
            [0.9453],
            [0.9593],
            [0.9734],
            [0.9875]])
    weights_file: tensor([[0.8609],
            [0.8750],
            [0.8890],
            [0.9031],
            [0.9172],
            [0.9312],
            [0.9453],
            [0.9593],
            [0.9734],
            [0.9875]])
    


```python
from sklearn.datasets import make_circles

x,y= make_circles(1000,noise=0.03,random_state=42)
len(x),len(y)
```




    (1000, 1000)




```python
print(x.shape,y.shape)
print(x[:5],y[:5])
```

    (1000, 2) (1000,)
    [[ 0.75424625  0.23148074]
     [-0.75615888  0.15325888]
     [-0.81539193  0.17328203]
     [-0.39373073  0.69288277]
     [ 0.44220765 -0.89672343]] [1 1 1 1 0]
    


```python
x= torch.from_numpy(x).type(torch.float)
y= torch.from_numpy(y).type(torch.float)
```


```python
plt.scatter(x[:,0],x[:,1],c=y)
```




    <matplotlib.collections.PathCollection at 0x26c29d6ec10>




    
![png](Learn%20Pytorch_files/Learn%20Pytorch_38_1.png)
    



```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
x =x.to(device)
y=y.to(device)
```


```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
```


```python
y_train.shape
```




    torch.Size([800])




```python
class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.main =nn.Sequential(
            nn.Linear(in_features=2,out_features=16),
            nn.Tanh(),
            nn.Linear(in_features=16,out_features=8),
            nn.Tanh(),
            nn.Linear(in_features=8,out_features=1)
        )
    def forward(self,x):
        return self.main(x)
```


```python
model_circle = CircleModel().to(device)
print(model_circle)
```

    CircleModel(
      (main): Sequential(
        (0): Linear(in_features=2, out_features=16, bias=True)
        (1): Tanh()
        (2): Linear(in_features=16, out_features=8, bias=True)
        (3): Tanh()
        (4): Linear(in_features=8, out_features=1, bias=True)
      )
    )
    


```python
model_circle.state_dict()
```




    OrderedDict([('main.0.weight',
                  tensor([[-0.1657,  0.6496],
                          [-0.1549,  0.1427],
                          [-0.3443,  0.4153],
                          [ 0.6233, -0.5188],
                          [ 0.6146,  0.1323],
                          [ 0.5224,  0.0958],
                          [ 0.3410, -0.0998],
                          [ 0.5451,  0.1045],
                          [-0.3301,  0.1802],
                          [-0.3258, -0.0829],
                          [-0.2872,  0.4691],
                          [-0.5582, -0.3260],
                          [-0.1997, -0.4252],
                          [ 0.0667, -0.6984],
                          [ 0.6386, -0.6007],
                          [ 0.5459,  0.1177]])),
                 ('main.0.bias',
                  tensor([-0.2296,  0.4370,  0.1102,  0.5713,  0.0773, -0.2230,  0.1900, -0.1918,
                           0.2976,  0.6313,  0.4087, -0.3091,  0.4082,  0.1265,  0.3591, -0.4310])),
                 ('main.2.weight',
                  tensor([[-0.2475, -0.0966, -0.1918,  0.2051,  0.0720,  0.1036,  0.0791, -0.0043,
                            0.1957, -0.1776,  0.0157, -0.1706,  0.0771, -0.0861,  0.0766, -0.0521],
                          [ 0.2073, -0.1482, -0.1491, -0.1491,  0.2249,  0.0833,  0.2406, -0.2063,
                           -0.2480, -0.1956, -0.1682,  0.1013,  0.0895,  0.2077, -0.1291, -0.1704],
                          [ 0.1326, -0.1011,  0.1517, -0.0593,  0.1430, -0.1942, -0.1262,  0.0762,
                            0.0529, -0.0637,  0.1490,  0.1700, -0.1813, -0.1335,  0.2289, -0.0844],
                          [-0.0886, -0.2419, -0.1432,  0.0625, -0.0330, -0.1815,  0.0059, -0.1708,
                           -0.2121, -0.1377, -0.2188, -0.1592,  0.2499,  0.0472,  0.0770, -0.2332],
                          [-0.1642, -0.0832,  0.0391, -0.2200, -0.1077, -0.1497,  0.0007, -0.0930,
                           -0.0173, -0.1694, -0.1716, -0.1459, -0.0856, -0.1973,  0.2096, -0.0496],
                          [ 0.2151,  0.0779, -0.2117,  0.1730, -0.0688, -0.0958, -0.2075, -0.2485,
                            0.0715, -0.0546,  0.0973, -0.2052,  0.1856, -0.1835, -0.0432,  0.0522],
                          [ 0.1291,  0.2018,  0.2277, -0.1982,  0.0629, -0.1075, -0.0274, -0.1871,
                            0.2277, -0.1835,  0.1336,  0.0879,  0.0812, -0.1352,  0.2272,  0.0549],
                          [ 0.0322, -0.2203,  0.1049, -0.0375, -0.1145,  0.2147,  0.0557, -0.1383,
                           -0.1265, -0.0119,  0.1396, -0.0639, -0.1426, -0.0856, -0.1868,  0.0892]])),
                 ('main.2.bias',
                  tensor([ 0.1935, -0.2354,  0.0581,  0.1291,  0.0453, -0.0890,  0.1305,  0.1314])),
                 ('main.4.weight',
                  tensor([[ 0.1322, -0.0621, -0.0936,  0.0378, -0.0625, -0.1054,  0.2260,  0.3038]])),
                 ('main.4.bias', tensor([-0.0350]))])




```python
loss_fn_3= nn.BCEWithLogitsLoss() #it consist of sigmoid layer it rquires logit input and it is better than BCELoss 
optimizer_3 = torch.optim.Adam(params=model_circle.parameters(),lr=0.01)
epoch_3 =150
```


```python
def accuracy1(y_true,y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = (correct/len(y_pred))*100
    return acc
```


```python
torch.manual_seed(42)
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epoch_3):
    model_circle.train()
    y_pred_logits = model_circle(x_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_pred_logits))
    loss = loss_fn_3(y_pred_logits,y_train)
    optimizer_3.zero_grad()
    loss.backward()
    optimizer_3.step()
    

    model_circle.eval()
    with torch.inference_mode():
        test_pred_logit = model_circle(x_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_pred_logit))
        test_loss = loss_fn_3(test_pred_logit,y_test)
        accuracy =  1-((torch.sum(torch.pow((y_test-test_pred),2)))/(torch.sum(torch.pow((y_test-y_test.mean()),2))))
        accuracy2 = accuracy1(y_test,test_pred)
        epoch_count.append(epoch)
        train_loss_values.append(loss.numpy())
        test_loss_values.append(test_loss.numpy())
    print(f"epoch:{epoch+1}/{epoch_3} | Loss: {loss} | Test loss:{test_loss} | test accuracy: {accuracy*100} | test accuracy with function:{accuracy2}")
```

    epoch:1/150 | Loss: 0.6931400895118713 | Test loss:0.6939513683319092 | test accuracy: -102.0 | test accuracy with function:49.5
    epoch:2/150 | Loss: 0.6925147771835327 | Test loss:0.6930491924285889 | test accuracy: -96.0 | test accuracy with function:51.0
    epoch:3/150 | Loss: 0.6920620799064636 | Test loss:0.6921751499176025 | test accuracy: -76.0 | test accuracy with function:56.00000000000001
    epoch:4/150 | Loss: 0.6915515661239624 | Test loss:0.6913607120513916 | test accuracy: -79.99999237060547 | test accuracy with function:55.00000000000001
    epoch:5/150 | Loss: 0.690993070602417 | Test loss:0.6906105279922485 | test accuracy: -67.99999237060547 | test accuracy with function:57.99999999999999
    epoch:6/150 | Loss: 0.6903893947601318 | Test loss:0.6898876428604126 | test accuracy: -82.00000762939453 | test accuracy with function:54.50000000000001
    epoch:7/150 | Loss: 0.6897338628768921 | Test loss:0.6891433000564575 | test accuracy: -94.00000762939453 | test accuracy with function:51.5
    epoch:8/150 | Loss: 0.6890203952789307 | Test loss:0.6883335709571838 | test accuracy: -88.0 | test accuracy with function:53.0
    epoch:9/150 | Loss: 0.6882431507110596 | Test loss:0.6874240040779114 | test accuracy: -58.000003814697266 | test accuracy with function:60.5
    epoch:10/150 | Loss: 0.6873953938484192 | Test loss:0.6863918900489807 | test accuracy: -38.0 | test accuracy with function:65.5
    epoch:11/150 | Loss: 0.6864678263664246 | Test loss:0.685228705406189 | test accuracy: -34.000003814697266 | test accuracy with function:66.5
    epoch:12/150 | Loss: 0.685452938079834 | Test loss:0.6839354038238525 | test accuracy: -38.0 | test accuracy with function:65.5
    epoch:13/150 | Loss: 0.68434739112854 | Test loss:0.6825225949287415 | test accuracy: -41.999996185302734 | test accuracy with function:64.5
    epoch:14/150 | Loss: 0.6831539273262024 | Test loss:0.6810056567192078 | test accuracy: -46.000003814697266 | test accuracy with function:63.5
    epoch:15/150 | Loss: 0.681877613067627 | Test loss:0.6793984770774841 | test accuracy: -44.00000762939453 | test accuracy with function:64.0
    epoch:16/150 | Loss: 0.6805201768875122 | Test loss:0.6777111291885376 | test accuracy: -46.000003814697266 | test accuracy with function:63.5
    epoch:17/150 | Loss: 0.6790785193443298 | Test loss:0.6759505271911621 | test accuracy: -50.0 | test accuracy with function:62.5
    epoch:18/150 | Loss: 0.677547812461853 | Test loss:0.6741254329681396 | test accuracy: -52.0 | test accuracy with function:62.0
    epoch:19/150 | Loss: 0.6759269833564758 | Test loss:0.6722479462623596 | test accuracy: -55.99999237060547 | test accuracy with function:61.0
    epoch:20/150 | Loss: 0.6742159128189087 | Test loss:0.6703264117240906 | test accuracy: -52.0 | test accuracy with function:62.0
    epoch:21/150 | Loss: 0.6724079251289368 | Test loss:0.6683590412139893 | test accuracy: -53.999996185302734 | test accuracy with function:61.5
    epoch:22/150 | Loss: 0.6704832315444946 | Test loss:0.6663371920585632 | test accuracy: -50.0 | test accuracy with function:62.5
    epoch:23/150 | Loss: 0.6684181094169617 | Test loss:0.6642476916313171 | test accuracy: -50.0 | test accuracy with function:62.5
    epoch:24/150 | Loss: 0.6661940813064575 | Test loss:0.6620693802833557 | test accuracy: -46.000003814697266 | test accuracy with function:63.5
    epoch:25/150 | Loss: 0.6637948751449585 | Test loss:0.6597704887390137 | test accuracy: -41.999996185302734 | test accuracy with function:64.5
    epoch:26/150 | Loss: 0.6611984968185425 | Test loss:0.6573249697685242 | test accuracy: -38.0 | test accuracy with function:65.5
    epoch:27/150 | Loss: 0.6583849191665649 | Test loss:0.6547274589538574 | test accuracy: -39.999996185302734 | test accuracy with function:65.0
    epoch:28/150 | Loss: 0.6553477644920349 | Test loss:0.6519769430160522 | test accuracy: -38.0 | test accuracy with function:65.5
    epoch:29/150 | Loss: 0.6520847082138062 | Test loss:0.6490573883056641 | test accuracy: -34.000003814697266 | test accuracy with function:66.5
    epoch:30/150 | Loss: 0.6485812664031982 | Test loss:0.6459401845932007 | test accuracy: -22.000003814697266 | test accuracy with function:69.5
    epoch:31/150 | Loss: 0.6448130011558533 | Test loss:0.6425877213478088 | test accuracy: -10.000001907348633 | test accuracy with function:72.5
    epoch:32/150 | Loss: 0.6407505869865417 | Test loss:0.638967752456665 | test accuracy: -3.9999961853027344 | test accuracy with function:74.0
    epoch:33/150 | Loss: 0.6363589763641357 | Test loss:0.6350609064102173 | test accuracy: -1.9999980926513672 | test accuracy with function:74.5
    epoch:34/150 | Loss: 0.6316025257110596 | Test loss:0.6308624148368835 | test accuracy: 0.0 | test accuracy with function:75.0
    epoch:35/150 | Loss: 0.6264572143554688 | Test loss:0.6263779997825623 | test accuracy: 4.000001907348633 | test accuracy with function:76.0
    epoch:36/150 | Loss: 0.620919406414032 | Test loss:0.6216199398040771 | test accuracy: 10.000001907348633 | test accuracy with function:77.5
    epoch:37/150 | Loss: 0.615004301071167 | Test loss:0.6166024804115295 | test accuracy: 18.0 | test accuracy with function:79.5
    epoch:38/150 | Loss: 0.6087366342544556 | Test loss:0.611336350440979 | test accuracy: 26.0 | test accuracy with function:81.5
    epoch:39/150 | Loss: 0.6021386384963989 | Test loss:0.6058146953582764 | test accuracy: 27.999996185302734 | test accuracy with function:82.0
    epoch:40/150 | Loss: 0.595216691493988 | Test loss:0.5999999046325684 | test accuracy: 36.0 | test accuracy with function:84.0
    epoch:41/150 | Loss: 0.5879562497138977 | Test loss:0.5938348770141602 | test accuracy: 36.0 | test accuracy with function:84.0
    epoch:42/150 | Loss: 0.5803319811820984 | Test loss:0.5872734785079956 | test accuracy: 52.0 | test accuracy with function:88.0
    epoch:43/150 | Loss: 0.5723246335983276 | Test loss:0.5803000330924988 | test accuracy: 62.0 | test accuracy with function:90.5
    epoch:44/150 | Loss: 0.5639225840568542 | Test loss:0.5729201436042786 | test accuracy: 66.0 | test accuracy with function:91.5
    epoch:45/150 | Loss: 0.5551138520240784 | Test loss:0.5651318430900574 | test accuracy: 74.0 | test accuracy with function:93.5
    epoch:46/150 | Loss: 0.5458831787109375 | Test loss:0.5568989515304565 | test accuracy: 74.0 | test accuracy with function:93.5
    epoch:47/150 | Loss: 0.5362164974212646 | Test loss:0.5481555461883545 | test accuracy: 80.0 | test accuracy with function:95.0
    epoch:48/150 | Loss: 0.5261050462722778 | Test loss:0.5388538837432861 | test accuracy: 84.0 | test accuracy with function:96.0
    epoch:49/150 | Loss: 0.515549898147583 | Test loss:0.529006838798523 | test accuracy: 84.0 | test accuracy with function:96.0
    epoch:50/150 | Loss: 0.5045652389526367 | Test loss:0.5186735987663269 | test accuracy: 88.0 | test accuracy with function:97.0
    epoch:51/150 | Loss: 0.4931720793247223 | Test loss:0.5079144835472107 | test accuracy: 90.0 | test accuracy with function:97.5
    epoch:52/150 | Loss: 0.48138943314552307 | Test loss:0.4967721998691559 | test accuracy: 90.0 | test accuracy with function:97.5
    epoch:53/150 | Loss: 0.4692279100418091 | Test loss:0.48527786135673523 | test accuracy: 90.0 | test accuracy with function:97.5
    epoch:54/150 | Loss: 0.4566907584667206 | Test loss:0.4734484553337097 | test accuracy: 90.0 | test accuracy with function:97.5
    epoch:55/150 | Loss: 0.443785160779953 | Test loss:0.46127015352249146 | test accuracy: 90.0 | test accuracy with function:97.5
    epoch:56/150 | Loss: 0.43052875995635986 | Test loss:0.44871044158935547 | test accuracy: 88.0 | test accuracy with function:97.0
    epoch:57/150 | Loss: 0.41694656014442444 | Test loss:0.4357629716396332 | test accuracy: 88.0 | test accuracy with function:97.0
    epoch:58/150 | Loss: 0.4030699133872986 | Test loss:0.42246609926223755 | test accuracy: 88.0 | test accuracy with function:97.0
    epoch:59/150 | Loss: 0.3889411985874176 | Test loss:0.40889012813568115 | test accuracy: 88.0 | test accuracy with function:97.0
    epoch:60/150 | Loss: 0.3746240735054016 | Test loss:0.3951466381549835 | test accuracy: 92.0 | test accuracy with function:98.0
    epoch:61/150 | Loss: 0.3602161407470703 | Test loss:0.3813823461532593 | test accuracy: 92.0 | test accuracy with function:98.0
    epoch:62/150 | Loss: 0.3458516299724579 | Test loss:0.367715984582901 | test accuracy: 92.0 | test accuracy with function:98.0
    epoch:63/150 | Loss: 0.33165985345840454 | Test loss:0.35420918464660645 | test accuracy: 90.0 | test accuracy with function:97.5
    epoch:64/150 | Loss: 0.3176742494106293 | Test loss:0.34088772535324097 | test accuracy: 90.0 | test accuracy with function:97.5
    epoch:65/150 | Loss: 0.303828626871109 | Test loss:0.3277079463005066 | test accuracy: 90.0 | test accuracy with function:97.5
    epoch:66/150 | Loss: 0.29006433486938477 | Test loss:0.31462156772613525 | test accuracy: 94.0 | test accuracy with function:98.5
    epoch:67/150 | Loss: 0.2763823866844177 | Test loss:0.30161958932876587 | test accuracy: 94.0 | test accuracy with function:98.5
    epoch:68/150 | Loss: 0.2628253698348999 | Test loss:0.28868478536605835 | test accuracy: 94.0 | test accuracy with function:98.5
    epoch:69/150 | Loss: 0.2494635432958603 | Test loss:0.27587947249412537 | test accuracy: 94.0 | test accuracy with function:98.5
    epoch:70/150 | Loss: 0.23638178408145905 | Test loss:0.2633240222930908 | test accuracy: 94.0 | test accuracy with function:98.5
    epoch:71/150 | Loss: 0.22366303205490112 | Test loss:0.2510540783405304 | test accuracy: 94.0 | test accuracy with function:98.5
    epoch:72/150 | Loss: 0.21137376129627228 | Test loss:0.2391394078731537 | test accuracy: 98.0 | test accuracy with function:99.5
    epoch:73/150 | Loss: 0.19955453276634216 | Test loss:0.22757388651371002 | test accuracy: 98.0 | test accuracy with function:99.5
    epoch:74/150 | Loss: 0.1882200390100479 | Test loss:0.21644365787506104 | test accuracy: 98.0 | test accuracy with function:99.5
    epoch:75/150 | Loss: 0.17737263441085815 | Test loss:0.2057601809501648 | test accuracy: 98.0 | test accuracy with function:99.5
    epoch:76/150 | Loss: 0.16701391339302063 | Test loss:0.19554708898067474 | test accuracy: 98.0 | test accuracy with function:99.5
    epoch:77/150 | Loss: 0.15715031325817108 | Test loss:0.18577855825424194 | test accuracy: 98.0 | test accuracy with function:99.5
    epoch:78/150 | Loss: 0.14779359102249146 | Test loss:0.17651645839214325 | test accuracy: 98.0 | test accuracy with function:99.5
    epoch:79/150 | Loss: 0.1389552354812622 | Test loss:0.16772843897342682 | test accuracy: 98.0 | test accuracy with function:99.5
    epoch:80/150 | Loss: 0.13064049184322357 | Test loss:0.1594715267419815 | test accuracy: 98.0 | test accuracy with function:99.5
    epoch:81/150 | Loss: 0.12284119427204132 | Test loss:0.15160132944583893 | test accuracy: 98.0 | test accuracy with function:99.5
    epoch:82/150 | Loss: 0.11553540080785751 | Test loss:0.1442495584487915 | test accuracy: 98.0 | test accuracy with function:99.5
    epoch:83/150 | Loss: 0.1086927205324173 | Test loss:0.1371479481458664 | test accuracy: 98.0 | test accuracy with function:99.5
    epoch:84/150 | Loss: 0.10228229314088821 | Test loss:0.1306362897157669 | test accuracy: 98.0 | test accuracy with function:99.5
    epoch:85/150 | Loss: 0.09627876430749893 | Test loss:0.12413454055786133 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:86/150 | Loss: 0.09066565334796906 | Test loss:0.11845722794532776 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:87/150 | Loss: 0.0854276493191719 | Test loss:0.11244222521781921 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:88/150 | Loss: 0.08055693656206131 | Test loss:0.10756019502878189 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:89/150 | Loss: 0.07602198421955109 | Test loss:0.10210659354925156 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:90/150 | Loss: 0.07180098444223404 | Test loss:0.09763112664222717 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:91/150 | Loss: 0.06786929070949554 | Test loss:0.09315565973520279 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:92/150 | Loss: 0.06422256678342819 | Test loss:0.08883758634328842 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:93/150 | Loss: 0.06083798408508301 | Test loss:0.08521461486816406 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:94/150 | Loss: 0.0576762817800045 | Test loss:0.08135421574115753 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:95/150 | Loss: 0.054723162204027176 | Test loss:0.07793522626161575 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:96/150 | Loss: 0.05198383331298828 | Test loss:0.07489866018295288 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:97/150 | Loss: 0.049442123621702194 | Test loss:0.07170987129211426 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:98/150 | Loss: 0.04706958308815956 | Test loss:0.06893854588270187 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:99/150 | Loss: 0.04486232250928879 | Test loss:0.06638669222593307 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:100/150 | Loss: 0.04281391203403473 | Test loss:0.06373469531536102 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:101/150 | Loss: 0.04089837521314621 | Test loss:0.061409153044223785 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:102/150 | Loss: 0.03910999745130539 | Test loss:0.05926983803510666 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:103/150 | Loss: 0.037445202469825745 | Test loss:0.057061225175857544 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:104/150 | Loss: 0.03588520362973213 | Test loss:0.05509267747402191 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:105/150 | Loss: 0.0344289131462574 | Test loss:0.05332217365503311 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:106/150 | Loss: 0.03306932374835014 | Test loss:0.051507383584976196 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:107/150 | Loss: 0.03179386258125305 | Test loss:0.049838900566101074 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:108/150 | Loss: 0.030602730810642242 | Test loss:0.04837571084499359 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:109/150 | Loss: 0.029484331607818604 | Test loss:0.04691300541162491 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:110/150 | Loss: 0.028434600681066513 | Test loss:0.04549022763967514 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:111/150 | Loss: 0.027449624612927437 | Test loss:0.044237490743398666 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:112/150 | Loss: 0.02652110531926155 | Test loss:0.04305022209882736 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:113/150 | Loss: 0.025648849084973335 | Test loss:0.04184504598379135 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:114/150 | Loss: 0.02482539415359497 | Test loss:0.04074030742049217 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:115/150 | Loss: 0.024049725383520126 | Test loss:0.03974473103880882 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:116/150 | Loss: 0.023317093029618263 | Test loss:0.03874924033880234 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:117/150 | Loss: 0.022624822333455086 | Test loss:0.037770919501781464 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:118/150 | Loss: 0.02197045274078846 | Test loss:0.036889176815748215 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:119/150 | Loss: 0.021350331604480743 | Test loss:0.036066439002752304 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:120/150 | Loss: 0.020763292908668518 | Test loss:0.035235658288002014 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:121/150 | Loss: 0.02020575851202011 | Test loss:0.034440621733665466 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:122/150 | Loss: 0.019677042961120605 | Test loss:0.03372129425406456 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:123/150 | Loss: 0.019173957407474518 | Test loss:0.033033277839422226 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:124/150 | Loss: 0.01869598589837551 | Test loss:0.032341085374355316 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:125/150 | Loss: 0.018240490928292274 | Test loss:0.03168700635433197 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:126/150 | Loss: 0.0178068857640028 | Test loss:0.031092651188373566 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:127/150 | Loss: 0.017392991110682487 | Test loss:0.03052053041756153 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:128/150 | Loss: 0.01699824258685112 | Test loss:0.029948413372039795 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:129/150 | Loss: 0.01662084087729454 | Test loss:0.02940782532095909 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:130/150 | Loss: 0.01626022905111313 | Test loss:0.028913138434290886 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:131/150 | Loss: 0.015914883464574814 | Test loss:0.028436774387955666 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:132/150 | Loss: 0.015584314242005348 | Test loss:0.02796146459877491 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:133/150 | Loss: 0.01526724360883236 | Test loss:0.0275085661560297 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:134/150 | Loss: 0.014963198453187943 | Test loss:0.027090273797512054 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:135/150 | Loss: 0.014671099372208118 | Test loss:0.026687415316700935 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:136/150 | Loss: 0.014390516094863415 | Test loss:0.02628369629383087 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:137/150 | Loss: 0.01412053033709526 | Test loss:0.02589240111410618 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:138/150 | Loss: 0.01386074535548687 | Test loss:0.02552701160311699 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:139/150 | Loss: 0.013610396534204483 | Test loss:0.025177454575896263 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:140/150 | Loss: 0.013369104824960232 | Test loss:0.02482864446938038 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:141/150 | Loss: 0.013136263005435467 | Test loss:0.02448602393269539 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:142/150 | Loss: 0.012911501340568066 | Test loss:0.024162428453564644 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:143/150 | Loss: 0.01269433181732893 | Test loss:0.0238551814109087 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:144/150 | Loss: 0.012484383769333363 | Test loss:0.023551760241389275 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:145/150 | Loss: 0.012281286530196667 | Test loss:0.02325103059411049 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:146/150 | Loss: 0.01208467222750187 | Test loss:0.022963300347328186 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:147/150 | Loss: 0.011894254945218563 | Test loss:0.022691287100315094 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:148/150 | Loss: 0.011709676124155521 | Test loss:0.022426104173064232 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:149/150 | Loss: 0.011530705727636814 | Test loss:0.022161763161420822 | test accuracy: 100.0 | test accuracy with function:100.0
    epoch:150/150 | Loss: 0.011357033625245094 | Test loss:0.021903786808252335 | test accuracy: 100.0 | test accuracy with function:100.0
    


```python
import requests
from pathlib import Path 

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary
```

    helper_functions.py already exists, skipping download
    


```python
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_circle, x_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_circle, x_test, y_test)
```


    
![png](Learn%20Pytorch_files/Learn%20Pytorch_49_0.png)
    



```python
# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend();
```


    
![png](Learn%20Pytorch_files/Learn%20Pytorch_50_0.png)
    



```python
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Set the hyperparameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# 1. Create multi-class data
x_blob, y_blob = make_blobs(n_samples=1000,
    n_features=NUM_FEATURES, # X features
    centers=NUM_CLASSES, # y labels 
    cluster_std=1.5, # give the clusters a little shake up (try changing this to 1.0, the default)
    random_state=RANDOM_SEED
)

# 2. Turn data into tensors
x_blob = torch.from_numpy(x_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)
print(x_blob[:5], y_blob[:5])

# 3. Split into train and test sets
x_blob_train, x_blob_test, y_blob_train, y_blob_test = train_test_split(x_blob,
    y_blob,
    test_size=0.2,
    random_state=RANDOM_SEED
)

# 4. Plot data
plt.figure(figsize=(10, 7))
plt.scatter(x_blob[:, 0], x_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
```

    tensor([[-8.4134,  6.9352],
            [-5.7665, -6.4312],
            [-6.0421, -6.7661],
            [ 3.9508,  0.6984],
            [ 4.2505, -0.2815]]) tensor([3, 2, 2, 1, 1])
    




    <matplotlib.collections.PathCollection at 0x26c44a95210>




    
![png](Learn%20Pytorch_files/Learn%20Pytorch_51_2.png)
    



```python
class BlobModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.main_2 = nn.Sequential(
            nn.Linear(in_features=2,out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16,out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8,out_features=4),
        )
    def forward(self,x):
        return self.main_2(x)
model_4 = BlobModel().to(device)
model_4
```




    BlobModel(
      (main_2): Sequential(
        (0): Linear(in_features=2, out_features=16, bias=True)
        (1): ReLU()
        (2): Linear(in_features=16, out_features=8, bias=True)
        (3): ReLU()
        (4): Linear(in_features=8, out_features=4, bias=True)
      )
    )




```python
loss_fn_4 = nn.CrossEntropyLoss()
optimizer_4 = torch.optim.Adam(params=model_4.parameters(),lr=0.001)
```


```python
torch.manual_seed(42)
train_loss_values = []
test_loss_values = []
epoch_count = []
epoch_4=100
for epoch in range(epoch_4):
    model_4.train()
    y_pred_logits = model_4(x_blob_train)
    y_pred = torch.softmax(y_pred_logits,dim=1).argmax(dim=1)
    loss = loss_fn_4(y_pred_logits,y_blob_train)
    optimizer_4.zero_grad()
    loss.backward()
    optimizer_4.step()
    

    model_4.eval()
    with torch.inference_mode():
        test_pred_logit = model_4(x_blob_test)
        test_pred = torch.softmax(test_pred_logit,dim=1).argmax(dim=1)
        test_loss = loss_fn_4(test_pred_logit,y_blob_test)
        accuracy =  1-((torch.sum(torch.pow((y_blob_test-test_pred),2)))/(torch.sum(torch.pow((y_blob_test-y_blob_test.type(torch.float).mean()),2))))
        accuracy2 = accuracy1(y_blob_test,test_pred)
        epoch_count.append(epoch)
        train_loss_values.append(loss.numpy())
        test_loss_values.append(test_loss.numpy())
    print(f"epoch:{epoch+1}/{epoch_4} | Loss: {loss} | Test loss:{test_loss} | test accuracy: {accuracy*100} | test accuracy with function:{accuracy2}")
```

    epoch:1/100 | Loss: 1.3347669839859009 | Test loss:1.3519423007965088 | test accuracy: -122.75138092041016 | test accuracy with function:43.5
    epoch:2/100 | Loss: 1.3237768411636353 | Test loss:1.3417487144470215 | test accuracy: -122.75138092041016 | test accuracy with function:43.5
    epoch:3/100 | Loss: 1.3130369186401367 | Test loss:1.3317506313323975 | test accuracy: -121.98326873779297 | test accuracy with function:43.5
    epoch:4/100 | Loss: 1.3025392293930054 | Test loss:1.321934461593628 | test accuracy: -124.28760528564453 | test accuracy with function:43.5
    epoch:5/100 | Loss: 1.2922616004943848 | Test loss:1.3122694492340088 | test accuracy: -123.51949310302734 | test accuracy with function:43.5
    epoch:6/100 | Loss: 1.2821741104125977 | Test loss:1.302842140197754 | test accuracy: -121.21517944335938 | test accuracy with function:44.0
    epoch:7/100 | Loss: 1.2723058462142944 | Test loss:1.293724536895752 | test accuracy: -113.53409576416016 | test accuracy with function:46.5
    epoch:8/100 | Loss: 1.262712836265564 | Test loss:1.2848795652389526 | test accuracy: -105.46894073486328 | test accuracy with function:48.5
    epoch:9/100 | Loss: 1.2534887790679932 | Test loss:1.2764689922332764 | test accuracy: -98.94002532958984 | test accuracy with function:51.0
    epoch:10/100 | Loss: 1.2446951866149902 | Test loss:1.2682077884674072 | test accuracy: -70.9040756225586 | test accuracy with function:60.5
    epoch:11/100 | Loss: 1.2359955310821533 | Test loss:1.2599843740463257 | test accuracy: -44.788394927978516 | test accuracy with function:69.0
    epoch:12/100 | Loss: 1.2273601293563843 | Test loss:1.2517989873886108 | test accuracy: -40.179752349853516 | test accuracy with function:70.5
    epoch:13/100 | Loss: 1.2187937498092651 | Test loss:1.2436860799789429 | test accuracy: -40.179752349853516 | test accuracy with function:70.5
    epoch:14/100 | Loss: 1.2103042602539062 | Test loss:1.2356467247009277 | test accuracy: -42.10002517700195 | test accuracy with function:70.5
    epoch:15/100 | Loss: 1.2018781900405884 | Test loss:1.2276560068130493 | test accuracy: -40.179752349853516 | test accuracy with function:70.5
    epoch:16/100 | Loss: 1.1935234069824219 | Test loss:1.219709873199463 | test accuracy: -35.187042236328125 | test accuracy with function:70.5
    epoch:17/100 | Loss: 1.1852341890335083 | Test loss:1.2118186950683594 | test accuracy: -27.12191390991211 | test accuracy with function:70.5
    epoch:18/100 | Loss: 1.177009105682373 | Test loss:1.203996181488037 | test accuracy: -22.897315979003906 | test accuracy with function:70.5
    epoch:19/100 | Loss: 1.1688182353973389 | Test loss:1.1962611675262451 | test accuracy: -18.67271614074707 | test accuracy with function:70.5
    epoch:20/100 | Loss: 1.1606544256210327 | Test loss:1.1886080503463745 | test accuracy: -11.759746551513672 | test accuracy with function:70.5
    epoch:21/100 | Loss: 1.1525541543960571 | Test loss:1.181009292602539 | test accuracy: -6.382989883422852 | test accuracy with function:70.5
    epoch:22/100 | Loss: 1.1444867849349976 | Test loss:1.1734753847122192 | test accuracy: -6.382989883422852 | test accuracy with function:70.5
    epoch:23/100 | Loss: 1.136478066444397 | Test loss:1.1659798622131348 | test accuracy: -4.078662395477295 | test accuracy with function:70.5
    epoch:24/100 | Loss: 1.128498911857605 | Test loss:1.1584982872009277 | test accuracy: 2.4502575397491455 | test accuracy with function:70.5
    epoch:25/100 | Loss: 1.1205474138259888 | Test loss:1.1510417461395264 | test accuracy: 7.827013969421387 | test accuracy with function:70.5
    epoch:26/100 | Loss: 1.1126563549041748 | Test loss:1.1436336040496826 | test accuracy: 10.131335258483887 | test accuracy with function:70.5
    epoch:27/100 | Loss: 1.1048187017440796 | Test loss:1.1362956762313843 | test accuracy: 13.5878267288208 | test accuracy with function:70.5
    epoch:28/100 | Loss: 1.0970135927200317 | Test loss:1.128977656364441 | test accuracy: 14.739984512329102 | test accuracy with function:70.5
    epoch:29/100 | Loss: 1.0892503261566162 | Test loss:1.1216977834701538 | test accuracy: 14.739984512329102 | test accuracy with function:70.5
    epoch:30/100 | Loss: 1.0815304517745972 | Test loss:1.1144235134124756 | test accuracy: 15.8921480178833 | test accuracy with function:70.5
    epoch:31/100 | Loss: 1.0738251209259033 | Test loss:1.1071473360061646 | test accuracy: 20.88485336303711 | test accuracy with function:70.5
    epoch:32/100 | Loss: 1.066131830215454 | Test loss:1.099866509437561 | test accuracy: 19.3486385345459 | test accuracy with function:71.0
    epoch:33/100 | Loss: 1.0584332942962646 | Test loss:1.0925770998001099 | test accuracy: 21.65296173095703 | test accuracy with function:71.0
    epoch:34/100 | Loss: 1.0507386922836304 | Test loss:1.0852596759796143 | test accuracy: 26.645666122436523 | test accuracy with function:71.0
    epoch:35/100 | Loss: 1.0430446863174438 | Test loss:1.0778965950012207 | test accuracy: 28.181880950927734 | test accuracy with function:71.5
    epoch:36/100 | Loss: 1.0353530645370483 | Test loss:1.0705456733703613 | test accuracy: 27.413772583007812 | test accuracy with function:71.5
    epoch:37/100 | Loss: 1.0276589393615723 | Test loss:1.0631864070892334 | test accuracy: 31.63836669921875 | test accuracy with function:71.5
    epoch:38/100 | Loss: 1.019974708557129 | Test loss:1.0558044910430908 | test accuracy: 32.790531158447266 | test accuracy with function:71.5
    epoch:39/100 | Loss: 1.0123012065887451 | Test loss:1.0484116077423096 | test accuracy: 33.942691802978516 | test accuracy with function:71.5
    epoch:40/100 | Loss: 1.0046411752700806 | Test loss:1.0410361289978027 | test accuracy: 40.087554931640625 | test accuracy with function:71.5
    epoch:41/100 | Loss: 0.9969956278800964 | Test loss:1.0336697101593018 | test accuracy: 43.15999221801758 | test accuracy with function:71.5
    epoch:42/100 | Loss: 0.9893600344657898 | Test loss:1.026307463645935 | test accuracy: 44.696205139160156 | test accuracy with function:72.0
    epoch:43/100 | Loss: 0.9817458987236023 | Test loss:1.018958568572998 | test accuracy: 44.696205139160156 | test accuracy with function:72.0
    epoch:44/100 | Loss: 0.9741553664207458 | Test loss:1.011613368988037 | test accuracy: 44.312156677246094 | test accuracy with function:71.5
    epoch:45/100 | Loss: 0.9665764570236206 | Test loss:1.004276990890503 | test accuracy: 47.38458251953125 | test accuracy with function:71.5
    epoch:46/100 | Loss: 0.9590113162994385 | Test loss:0.9969477653503418 | test accuracy: 50.4570198059082 | test accuracy with function:71.5
    epoch:47/100 | Loss: 0.9514656662940979 | Test loss:0.9896343946456909 | test accuracy: 50.4570198059082 | test accuracy with function:71.5
    epoch:48/100 | Loss: 0.9439405202865601 | Test loss:0.9823325872421265 | test accuracy: 53.52945327758789 | test accuracy with function:71.5
    epoch:49/100 | Loss: 0.9364201426506042 | Test loss:0.9750409126281738 | test accuracy: 53.52945327758789 | test accuracy with function:71.5
    epoch:50/100 | Loss: 0.9289138913154602 | Test loss:0.9677443504333496 | test accuracy: 53.52945327758789 | test accuracy with function:71.5
    epoch:51/100 | Loss: 0.9214264154434204 | Test loss:0.9604471325874329 | test accuracy: 59.67432403564453 | test accuracy with function:71.5
    epoch:52/100 | Loss: 0.9139521718025208 | Test loss:0.9531406164169312 | test accuracy: 59.67432403564453 | test accuracy with function:71.5
    epoch:53/100 | Loss: 0.9064981937408447 | Test loss:0.9458321332931519 | test accuracy: 60.058372497558594 | test accuracy with function:72.0
    epoch:54/100 | Loss: 0.8990603089332581 | Test loss:0.9385225772857666 | test accuracy: 63.13080596923828 | test accuracy with function:72.0
    epoch:55/100 | Loss: 0.8916337490081787 | Test loss:0.9312131404876709 | test accuracy: 63.13080596923828 | test accuracy with function:72.0
    epoch:56/100 | Loss: 0.8842250108718872 | Test loss:0.9239063858985901 | test accuracy: 63.13080596923828 | test accuracy with function:72.0
    epoch:57/100 | Loss: 0.8768315315246582 | Test loss:0.9165951013565063 | test accuracy: 63.13080596923828 | test accuracy with function:72.0
    epoch:58/100 | Loss: 0.8694468140602112 | Test loss:0.9092810750007629 | test accuracy: 63.13080596923828 | test accuracy with function:72.0
    epoch:59/100 | Loss: 0.8620730042457581 | Test loss:0.9019532203674316 | test accuracy: 63.13080596923828 | test accuracy with function:72.0
    epoch:60/100 | Loss: 0.8547126054763794 | Test loss:0.8946127891540527 | test accuracy: 63.13080596923828 | test accuracy with function:72.0
    epoch:61/100 | Loss: 0.8473605513572693 | Test loss:0.8872557878494263 | test accuracy: 60.058372497558594 | test accuracy with function:72.0
    epoch:62/100 | Loss: 0.8400108218193054 | Test loss:0.8798860311508179 | test accuracy: 60.44242477416992 | test accuracy with function:72.5
    epoch:63/100 | Loss: 0.8326584696769714 | Test loss:0.8724998235702515 | test accuracy: 60.826480865478516 | test accuracy with function:73.0
    epoch:64/100 | Loss: 0.8252938985824585 | Test loss:0.8650994896888733 | test accuracy: 60.826480865478516 | test accuracy with function:73.0
    epoch:65/100 | Loss: 0.817911684513092 | Test loss:0.857676088809967 | test accuracy: 60.826480865478516 | test accuracy with function:73.0
    epoch:66/100 | Loss: 0.8105045557022095 | Test loss:0.8502216935157776 | test accuracy: 61.59458923339844 | test accuracy with function:74.0
    epoch:67/100 | Loss: 0.8030705451965332 | Test loss:0.8427350521087646 | test accuracy: 62.74674987792969 | test accuracy with function:75.5
    epoch:68/100 | Loss: 0.7956158518791199 | Test loss:0.8352128863334656 | test accuracy: 64.28296661376953 | test accuracy with function:77.5
    epoch:69/100 | Loss: 0.7881365418434143 | Test loss:0.8276448249816895 | test accuracy: 65.05107879638672 | test accuracy with function:78.5
    epoch:70/100 | Loss: 0.7806348204612732 | Test loss:0.8200337290763855 | test accuracy: 66.20323944091797 | test accuracy with function:80.0
    epoch:71/100 | Loss: 0.7731053233146667 | Test loss:0.8123689889907837 | test accuracy: 67.73944854736328 | test accuracy with function:82.0
    epoch:72/100 | Loss: 0.7655448317527771 | Test loss:0.804657518863678 | test accuracy: 68.12350463867188 | test accuracy with function:82.5
    epoch:73/100 | Loss: 0.7579538226127625 | Test loss:0.7968991994857788 | test accuracy: 70.4278335571289 | test accuracy with function:85.5
    epoch:74/100 | Loss: 0.7503319382667542 | Test loss:0.7890810370445251 | test accuracy: 73.1162109375 | test accuracy with function:89.0
    epoch:75/100 | Loss: 0.7426777482032776 | Test loss:0.7812150716781616 | test accuracy: 73.88432312011719 | test accuracy with function:90.0
    epoch:76/100 | Loss: 0.7349932193756104 | Test loss:0.7732882499694824 | test accuracy: 74.26837921142578 | test accuracy with function:90.5
    epoch:77/100 | Loss: 0.7272834181785583 | Test loss:0.7652978301048279 | test accuracy: 76.18864440917969 | test accuracy with function:93.0
    epoch:78/100 | Loss: 0.719540536403656 | Test loss:0.7572629451751709 | test accuracy: 77.72486114501953 | test accuracy with function:95.0
    epoch:79/100 | Loss: 0.7117649912834167 | Test loss:0.7491822242736816 | test accuracy: 77.72486114501953 | test accuracy with function:95.0
    epoch:80/100 | Loss: 0.7039593458175659 | Test loss:0.7410523891448975 | test accuracy: 78.10891723632812 | test accuracy with function:95.5
    epoch:81/100 | Loss: 0.6961202025413513 | Test loss:0.7328564524650574 | test accuracy: 78.49296569824219 | test accuracy with function:96.0
    epoch:82/100 | Loss: 0.6882340312004089 | Test loss:0.724592387676239 | test accuracy: 79.26107788085938 | test accuracy with function:97.0
    epoch:83/100 | Loss: 0.6803136467933655 | Test loss:0.7162806987762451 | test accuracy: 79.26107788085938 | test accuracy with function:97.0
    epoch:84/100 | Loss: 0.6723549365997314 | Test loss:0.7079299688339233 | test accuracy: 79.26107788085938 | test accuracy with function:97.0
    epoch:85/100 | Loss: 0.6643663644790649 | Test loss:0.6995298266410828 | test accuracy: 82.71755981445312 | test accuracy with function:97.5
    epoch:86/100 | Loss: 0.656355082988739 | Test loss:0.6910710334777832 | test accuracy: 82.71755981445312 | test accuracy with function:97.5
    epoch:87/100 | Loss: 0.6483111381530762 | Test loss:0.6825709342956543 | test accuracy: 82.71755981445312 | test accuracy with function:97.5
    epoch:88/100 | Loss: 0.6402342915534973 | Test loss:0.6740245819091797 | test accuracy: 82.71755981445312 | test accuracy with function:97.5
    epoch:89/100 | Loss: 0.6321315765380859 | Test loss:0.6654422283172607 | test accuracy: 82.71755981445312 | test accuracy with function:97.5
    epoch:90/100 | Loss: 0.6240090727806091 | Test loss:0.6568109393119812 | test accuracy: 82.71755981445312 | test accuracy with function:97.5
    epoch:91/100 | Loss: 0.6158653497695923 | Test loss:0.6481366157531738 | test accuracy: 82.71755981445312 | test accuracy with function:97.5
    epoch:92/100 | Loss: 0.6077033281326294 | Test loss:0.6394426822662354 | test accuracy: 82.71755981445312 | test accuracy with function:97.5
    epoch:93/100 | Loss: 0.5995297431945801 | Test loss:0.630720317363739 | test accuracy: 82.71755981445312 | test accuracy with function:97.5
    epoch:94/100 | Loss: 0.591342568397522 | Test loss:0.6219605803489685 | test accuracy: 86.1740493774414 | test accuracy with function:98.0
    epoch:95/100 | Loss: 0.58315509557724 | Test loss:0.6131662726402283 | test accuracy: 86.1740493774414 | test accuracy with function:98.0
    epoch:96/100 | Loss: 0.5749756693840027 | Test loss:0.6043570041656494 | test accuracy: 86.1740493774414 | test accuracy with function:98.0
    epoch:97/100 | Loss: 0.5667973756790161 | Test loss:0.595538318157196 | test accuracy: 86.1740493774414 | test accuracy with function:98.0
    epoch:98/100 | Loss: 0.5586123466491699 | Test loss:0.5867236256599426 | test accuracy: 86.1740493774414 | test accuracy with function:98.0
    epoch:99/100 | Loss: 0.5504167079925537 | Test loss:0.577913224697113 | test accuracy: 89.63053894042969 | test accuracy with function:98.5
    epoch:100/100 | Loss: 0.5422134399414062 | Test loss:0.5690976977348328 | test accuracy: 89.63053894042969 | test accuracy with function:98.5
    


```python
model_4.state_dict()
```




    OrderedDict([('main_2.0.weight',
                  tensor([[ 0.6425,  0.6970],
                          [-0.2694,  0.6185],
                          [-0.1574,  0.2562],
                          [-0.3688,  0.5236],
                          [ 0.5435, -0.6099],
                          [ 0.6374,  0.2530],
                          [ 0.6117,  0.1812],
                          [ 0.4303, -0.0088],
                          [ 0.6392,  0.0658],
                          [-0.4168,  0.0847],
                          [-0.3403, -0.0240],
                          [-0.3401,  0.3864],
                          [-0.6328, -0.4127],
                          [-0.2809, -0.5065],
                          [ 0.1361, -0.6330],
                          [ 0.6822, -0.6237]])),
                 ('main_2.0.bias',
                  tensor([ 0.6507,  0.0741, -0.1179,  0.4110,  0.0680,  0.6869,  0.1691, -0.1310,
                           0.2857, -0.1227,  0.3054,  0.5535,  0.4864, -0.2268,  0.3509,  0.2312])),
                 ('main_2.2.weight',
                  tensor([[ 2.1271e-01, -6.0262e-02, -1.4369e-01,  1.5537e-02, -1.1711e-01,
                            2.8857e-01,  1.5574e-01,  1.9014e-01,  1.6167e-01, -1.0301e-02,
                            1.8015e-01, -7.9504e-02, -8.4384e-04, -1.8729e-01,  6.7510e-02,
                           -4.1128e-03],
                          [ 1.5700e-01, -6.6076e-03,  2.2086e-01, -1.1193e-01, -7.2381e-02,
                           -6.5977e-02,  3.0920e-01,  1.7276e-01,  3.2477e-01, -2.2590e-01,
                           -3.0294e-01, -1.5865e-01, -2.0981e-01,  5.9246e-02,  1.8109e-01,
                            2.9108e-01],
                          [-1.4503e-01, -7.6439e-02,  2.3011e-01, -4.1214e-03,  2.6661e-01,
                           -1.3583e-01,  8.4522e-02, -1.9424e-01, -1.2616e-01,  1.8759e-01,
                            1.4975e-01,  3.8511e-02,  2.4979e-01,  2.7064e-01, -8.8740e-02,
                           -3.7724e-02],
                          [ 1.9759e-01, -1.1455e-01, -1.0317e-01, -2.7056e-01, -1.7220e-01,
                            3.1444e-02, -6.4204e-02, -2.1302e-01, -2.5089e-02, -1.7077e-01,
                           -2.1210e-01, -1.6499e-01, -2.1880e-01, -1.5918e-01,  2.2642e-01,
                            1.8005e-02],
                          [ 7.7040e-02, -2.3317e-01, -1.6419e-01, -8.3214e-02,  3.9093e-02,
                           -2.1998e-01, -1.0772e-01, -1.4967e-01,  6.9281e-04, -9.3026e-02,
                           -1.7324e-02, -1.6941e-01, -1.7160e-01, -1.4585e-01, -8.5574e-02,
                           -1.9732e-01],
                          [ 3.1217e-01,  4.1516e-02,  2.8962e-01,  1.6834e-01, -2.1003e-01,
                            2.6563e-01,  6.9019e-03, -9.1836e-02, -1.5751e-01, -3.4379e-01,
                           -4.1039e-03,  1.7877e-02,  2.6083e-02, -2.3833e-01,  1.5349e-01,
                           -2.2153e-01],
                          [-1.4751e-01,  1.8983e-01,  2.6990e-01,  3.3771e-01,  1.2060e-01,
                           -3.2244e-01, -6.4506e-02, -1.3733e-01, -1.4857e-01, -1.4509e-01,
                            1.8429e-01, -4.4460e-02,  5.9719e-02, -2.4739e-03, -2.0013e-04,
                           -2.2124e-01],
                          [ 3.2800e-01, -2.9581e-02, -6.7243e-02, -3.0881e-01,  1.9296e-01,
                            6.0290e-02, -1.6010e-02,  3.1893e-01,  1.5486e-01, -2.1220e-01,
                           -1.9851e-01, -9.6607e-02,  1.0150e-01, -1.0187e-01, -5.2029e-02,
                            1.0198e-02]])),
                 ('main_2.2.bias',
                  tensor([-0.1105,  0.1708,  0.2907, -0.2653,  0.0581,  0.1790, -0.0201,  0.0099])),
                 ('main_2.4.weight',
                  tensor([[ 0.0900,  0.0944,  0.0132, -0.0875, -0.0936,  0.1581, -0.1004, -0.2073],
                          [ 0.3199,  0.3941, -0.1484, -0.0545,  0.0052, -0.1392, -0.0175,  0.1942],
                          [-0.4005, -0.2029,  0.3997,  0.1157, -0.0176, -0.3012, -0.2098, -0.3827],
                          [-0.2107,  0.0266,  0.2193,  0.1386, -0.3125, -0.2734,  0.0773,  0.2489]])),
                 ('main_2.4.bias', tensor([ 0.0484, -0.2102,  0.2172, -0.0177]))])




```python
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_4, x_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_4, x_blob_test, y_blob_test)
```


    
![png](Learn%20Pytorch_files/Learn%20Pytorch_56_0.png)
    



```python
try:
    from torchmetrics import Accuracy
except:
    !pip install torchmetrics==0.9.3 # this is the version we're using in this notebook (later versions exist here: https://torchmetrics.readthedocs.io/en/stable/generated/CHANGELOG.html#changelog)
    from torchmetrics import Accuracy

# Setup metric and make sure it's on the target device
torchmetrics_accuracy = Accuracy(task='multiclass', num_classes=4).to(device)

# Calculate accuracy
torchmetrics_accuracy(test_pred, y_blob_test)
```




    tensor(0.9850)




```python
import numpy as np
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
x = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  x[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data
plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()
```


    
![png](Learn%20Pytorch_files/Learn%20Pytorch_58_0.png)
    



```python
x= torch.from_numpy(x).type(torch.float)
y= torch.from_numpy(y).type(torch.float)
```


```python
class SpiralModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.main_2 = nn.Sequential(
            nn.Linear(in_features=2,out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64,out_features=32),
            nn.Tanh(),
            nn.Linear(in_features=32,out_features=16),
            nn.Tanh(),
            nn.Linear(in_features=16,out_features=3),
        )
    def forward(self,x):
        return self.main_2(x)
model_5 = SpiralModel().to(device)
model_5
```




    SpiralModel(
      (main_2): Sequential(
        (0): Linear(in_features=2, out_features=64, bias=True)
        (1): Tanh()
        (2): Linear(in_features=64, out_features=32, bias=True)
        (3): Tanh()
        (4): Linear(in_features=32, out_features=16, bias=True)
        (5): Tanh()
        (6): Linear(in_features=16, out_features=3, bias=True)
      )
    )




```python
loss_fn_5 = nn.CrossEntropyLoss()
optimizer_5 = torch.optim.Adam(params=model_5.parameters(),lr=0.001)
```


```python
x_spiral_train, x_spiral_test, y_spiral_train, y_spiral_test = train_test_split(x,
    y,
    test_size=0.2,
    random_state=RANDOM_SEED
)
```


```python
y_spiral_train.type(torch.long)
```




    tensor([2, 0, 0, 1, 1, 0, 0, 2, 0, 1, 1, 1, 2, 1, 1, 2, 0, 0, 0, 2, 1, 1, 0, 1,
            0, 1, 1, 0, 0, 0, 1, 1, 0, 2, 0, 0, 1, 1, 2, 0, 2, 0, 0, 1, 2, 2, 1, 0,
            1, 0, 1, 2, 0, 2, 2, 1, 1, 2, 1, 2, 0, 1, 1, 1, 1, 2, 0, 1, 1, 2, 1, 1,
            1, 1, 2, 2, 1, 1, 1, 0, 1, 1, 1, 1, 2, 2, 1, 0, 0, 0, 2, 1, 2, 1, 2, 2,
            2, 0, 1, 0, 2, 1, 0, 0, 0, 1, 2, 2, 1, 0, 0, 2, 2, 0, 0, 2, 1, 0, 2, 1,
            2, 0, 2, 2, 1, 2, 2, 1, 0, 0, 0, 1, 2, 2, 2, 0, 2, 2, 0, 2, 0, 1, 0, 2,
            1, 2, 2, 1, 0, 1, 1, 2, 0, 0, 2, 0, 0, 2, 1, 0, 1, 2, 2, 1, 0, 2, 1, 0,
            2, 2, 0, 2, 2, 2, 1, 0, 0, 0, 2, 2, 2, 2, 1, 0, 2, 1, 1, 2, 2, 0, 1, 0,
            1, 0, 1, 1, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 2, 1, 2, 2, 0, 0, 1, 1, 2, 1,
            1, 0, 0, 2, 2, 0, 1, 2, 1, 2, 1, 1, 1, 0, 0, 2, 1, 2, 0, 1, 0, 1, 2, 1])




```python
torch.manual_seed(42)
train_loss_values = []
test_loss_values = []
epoch_count = []
epoch_5=220
for epoch in range(epoch_5):
    model_5.train()
    y_pred_logits = model_5(x_spiral_train)
    y_pred = torch.softmax(y_pred_logits,dim=1).argmax(dim=1)
    loss = loss_fn_5(y_pred_logits,y_spiral_train.type(torch.long))
    optimizer_5.zero_grad()
    loss.backward()
    optimizer_5.step()
    

    model_5.eval()
    with torch.inference_mode():
        test_pred_logit = model_5(x_spiral_test)
        test_pred = torch.softmax(test_pred_logit,dim=1).argmax(dim=1)
        test_loss = loss_fn_5(test_pred_logit,y_spiral_test.type(torch.long))
        accuracy =  1-((torch.sum(torch.pow((y_spiral_test-test_pred),2)))/(torch.sum(torch.pow((y_spiral_test-y_spiral_test.mean()),2))))
        accuracy2 = accuracy1(y_spiral_test,test_pred)
        epoch_count.append(epoch)
        train_loss_values.append(loss.numpy())
        test_loss_values.append(test_loss.numpy())
    print(f"epoch:{epoch+1}/{epoch_5} | Loss: {loss} | Test loss:{test_loss} | test accuracy: {accuracy*100} | test accuracy with function:{accuracy2}")
```

    epoch:1/20 | Loss: 0.13893774151802063 | Test loss:0.21015053987503052 | test accuracy: 88.63636016845703 | test accuracy with function:96.66666666666667
    epoch:2/20 | Loss: 0.13769122958183289 | Test loss:0.20938915014266968 | test accuracy: 88.63636016845703 | test accuracy with function:96.66666666666667
    epoch:3/20 | Loss: 0.13646255433559418 | Test loss:0.20863984525203705 | test accuracy: 88.63636016845703 | test accuracy with function:96.66666666666667
    epoch:4/20 | Loss: 0.13525137305259705 | Test loss:0.20790132880210876 | test accuracy: 88.63636016845703 | test accuracy with function:96.66666666666667
    epoch:5/20 | Loss: 0.13405722379684448 | Test loss:0.2071724385023117 | test accuracy: 88.63636016845703 | test accuracy with function:96.66666666666667
    epoch:6/20 | Loss: 0.13287971913814545 | Test loss:0.20645183324813843 | test accuracy: 88.63636016845703 | test accuracy with function:96.66666666666667
    epoch:7/20 | Loss: 0.13171851634979248 | Test loss:0.20573821663856506 | test accuracy: 88.63636016845703 | test accuracy with function:96.66666666666667
    epoch:8/20 | Loss: 0.13057324290275574 | Test loss:0.2050304412841797 | test accuracy: 88.63636016845703 | test accuracy with function:96.66666666666667
    epoch:9/20 | Loss: 0.12944351136684418 | Test loss:0.20432718098163605 | test accuracy: 88.63636016845703 | test accuracy with function:96.66666666666667
    epoch:10/20 | Loss: 0.12832900881767273 | Test loss:0.20362725853919983 | test accuracy: 88.63636016845703 | test accuracy with function:96.66666666666667
    epoch:11/20 | Loss: 0.12722936272621155 | Test loss:0.2029295563697815 | test accuracy: 88.63636016845703 | test accuracy with function:96.66666666666667
    epoch:12/20 | Loss: 0.12614421546459198 | Test loss:0.20223288238048553 | test accuracy: 88.63636016845703 | test accuracy with function:96.66666666666667
    epoch:13/20 | Loss: 0.12507326900959015 | Test loss:0.20153623819351196 | test accuracy: 88.63636016845703 | test accuracy with function:96.66666666666667
    epoch:14/20 | Loss: 0.12401613593101501 | Test loss:0.20083855092525482 | test accuracy: 88.63636016845703 | test accuracy with function:96.66666666666667
    epoch:15/20 | Loss: 0.12297255545854568 | Test loss:0.2001388818025589 | test accuracy: 88.63636016845703 | test accuracy with function:96.66666666666667
    epoch:16/20 | Loss: 0.1219421848654747 | Test loss:0.19943636655807495 | test accuracy: 88.63636016845703 | test accuracy with function:96.66666666666667
    epoch:17/20 | Loss: 0.12092471122741699 | Test loss:0.19873015582561493 | test accuracy: 88.63636016845703 | test accuracy with function:96.66666666666667
    epoch:18/20 | Loss: 0.11991982907056808 | Test loss:0.19801950454711914 | test accuracy: 88.63636016845703 | test accuracy with function:96.66666666666667
    epoch:19/20 | Loss: 0.1189272552728653 | Test loss:0.19730378687381744 | test accuracy: 88.63636016845703 | test accuracy with function:96.66666666666667
    epoch:20/20 | Loss: 0.11794672161340714 | Test loss:0.19658245146274567 | test accuracy: 88.63636016845703 | test accuracy with function:96.66666666666667
    


```python
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_5, x_spiral_train, y_spiral_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_5, x_spiral_test, y_spiral_test)
```


    
![png](Learn%20Pytorch_files/Learn%20Pytorch_65_0.png)
    



```python
try:
    from torchmetrics import Accuracy
except:
    !pip install torchmetrics==0.9.3 # this is the version we're using in this notebook (later versions exist here: https://torchmetrics.readthedocs.io/en/stable/generated/CHANGELOG.html#changelog)
    from torchmetrics import Accuracy

# Setup metric and make sure it's on the target device
torchmetrics_accuracy = Accuracy(task='multiclass', num_classes=4).to(device)

# Calculate accuracy
torchmetrics_accuracy(test_pred, y_spiral_test)
```




    tensor(0.9667)




```python
y_spiral_test.type(torch.long)
```




    tensor([2, 2, 1, 0, 2, 2, 1, 1, 0, 1, 2, 0, 2, 0, 1, 2, 2, 2, 1, 1, 0, 1, 2, 2,
            1, 0, 2, 2, 1, 2, 0, 1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 2,
            1, 2, 1, 1, 2, 0, 2, 2, 0, 0, 1, 0])




```python
from sklearn.metrics import classification_report

y_true = y_spiral_test.type(torch.long).numpy()  # Assuming y_spiral_test is ground truth
y_pred = test_pred.numpy()  # Assuming test_pred is predicted labels

report = classification_report(y_true=y_true, y_pred=y_pred, labels=['class 0', 'class 1', 'class 2'],zero_division=0)

print(report)

```

                  precision    recall  f1-score   support
    
         class 0       0.00      0.00      0.00         0
         class 1       0.00      0.00      0.00         0
         class 2       0.00      0.00      0.00         0
    
       micro avg       0.00      0.00      0.00         0
       macro avg       0.00      0.00      0.00         0
    weighted avg       0.00      0.00      0.00         0
    
    


```python
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix

num_classes = 3

acc_per_class = MulticlassAccuracy(num_classes=num_classes, average=None)
confmat = MulticlassConfusionMatrix(num_classes=num_classes)

preds = test_pred.type(torch.long)
target= y_spiral_test.type(torch.long)

# plot single value
acc_per_class.update(preds, target)
confmat.update(preds, target)
fig1, ax1 = acc_per_class.plot()
fig2, ax2 = confmat.plot()
```


    
![png](Learn%20Pytorch_files/Learn%20Pytorch_69_0.png)
    



    
![png](Learn%20Pytorch_files/Learn%20Pytorch_69_1.png)
    


Computer Vision
https://www.learnpytorch.io/03_pytorch_computer_vision/


```python
# Import PyTorch
import torch
from torch import nn

# Import torchvision 
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

# Import matplotlib for visualization
import matplotlib.pyplot as plt

# Check versions
# Note: your PyTorch version shouldn't be lower than 1.10.0 and torchvision version shouldn't be lower than 0.11
print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")
```

    PyTorch version: 2.1.2+cpu
    torchvision version: 0.16.2+cpu
    


```python
# Setup training data
train_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=True, # get training data
    download=True, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)

# Setup testing data
test_data = datasets.FashionMNIST(
    root="data",
    train=False, # get test data
    download=True,
    transform=ToTensor()
)
```


```python
# See first training sample
image, label = train_data[10]
# See classes
class_names = train_data.classes
print(image,label)
# image.squeeze().shape
plt.imshow(image.permute(1,2,0),cmap='gray')
plt.title(class_names[label])
```

    tensor([[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0431,
              0.5569, 0.7843, 0.4157, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
              0.0000, 0.0000, 0.3333, 0.7255, 0.4392, 0.0000, 0.0000, 0.0000,
              0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5961, 0.8392,
              0.8510, 0.7608, 0.9255, 0.8471, 0.7333, 0.5843, 0.5294, 0.6000,
              0.8275, 0.8510, 0.9059, 0.8039, 0.8510, 0.7373, 0.1333, 0.0000,
              0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2588, 0.7255, 0.6510,
              0.7059, 0.7098, 0.7451, 0.8275, 0.8667, 0.7725, 0.5725, 0.7765,
              0.8078, 0.7490, 0.6588, 0.7451, 0.6745, 0.7373, 0.6863, 0.0000,
              0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5294, 0.6000, 0.6275,
              0.6863, 0.7059, 0.6667, 0.7294, 0.7333, 0.7451, 0.7373, 0.7451,
              0.7333, 0.6824, 0.7647, 0.7255, 0.6824, 0.6314, 0.6863, 0.2314,
              0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.6314, 0.5765, 0.6275,
              0.6667, 0.6980, 0.6941, 0.7059, 0.6588, 0.6784, 0.6824, 0.6706,
              0.7255, 0.7216, 0.7255, 0.6745, 0.6706, 0.6431, 0.6824, 0.4706,
              0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0078, 0.6863, 0.5725, 0.5686,
              0.6588, 0.6980, 0.7098, 0.7255, 0.7059, 0.7216, 0.6980, 0.7020,
              0.7333, 0.7490, 0.7569, 0.7451, 0.7098, 0.6706, 0.6745, 0.6196,
              0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.1373, 0.6941, 0.6078, 0.5490,
              0.5922, 0.6745, 0.7490, 0.7333, 0.7294, 0.7333, 0.7294, 0.7333,
              0.7137, 0.7490, 0.7608, 0.7373, 0.7059, 0.6314, 0.6314, 0.7255,
              0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.2314, 0.6667, 0.6000, 0.5529,
              0.4706, 0.6039, 0.6275, 0.6314, 0.6745, 0.6588, 0.6510, 0.6314,
              0.6471, 0.6745, 0.6667, 0.6431, 0.5451, 0.5843, 0.6353, 0.6510,
              0.0824, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.3098, 0.5686, 0.6275, 0.8392,
              0.4824, 0.5020, 0.6000, 0.6275, 0.6431, 0.6196, 0.6157, 0.6039,
              0.6078, 0.6667, 0.6471, 0.5529, 0.7647, 0.7569, 0.5961, 0.6510,
              0.2392, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.3922, 0.6157, 0.8824, 0.9608,
              0.6863, 0.4431, 0.6824, 0.6196, 0.6196, 0.6275, 0.6078, 0.6275,
              0.6431, 0.6980, 0.7373, 0.5294, 0.7255, 0.9412, 0.7882, 0.6745,
              0.4235, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1216, 0.6824, 0.1098,
              0.4941, 0.6000, 0.6510, 0.5961, 0.6196, 0.6196, 0.6275, 0.6314,
              0.6157, 0.6588, 0.7490, 0.7373, 0.0706, 0.5176, 0.6235, 0.0275,
              0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
              0.3216, 0.7333, 0.6235, 0.6000, 0.6157, 0.6196, 0.6353, 0.6431,
              0.6431, 0.6039, 0.7333, 0.7451, 0.0000, 0.0000, 0.0000, 0.0000,
              0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0039, 0.0118, 0.0196, 0.0000,
              0.1451, 0.6863, 0.6196, 0.6078, 0.6353, 0.6196, 0.6275, 0.6353,
              0.6471, 0.6000, 0.6941, 0.8039, 0.0000, 0.0000, 0.0118, 0.0118,
              0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0039, 0.0000,
              0.0980, 0.6863, 0.5961, 0.6275, 0.6196, 0.6314, 0.6275, 0.6431,
              0.6431, 0.6314, 0.6510, 0.7843, 0.0000, 0.0000, 0.0039, 0.0000,
              0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0157, 0.0000,
              0.1176, 0.6706, 0.5765, 0.6431, 0.6078, 0.6471, 0.6314, 0.6471,
              0.6353, 0.6667, 0.6431, 0.6353, 0.0000, 0.0000, 0.0078, 0.0000,
              0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0157, 0.0000,
              0.2235, 0.6510, 0.6078, 0.6431, 0.6510, 0.6314, 0.6314, 0.6431,
              0.6549, 0.6471, 0.6471, 0.6353, 0.1098, 0.0000, 0.0118, 0.0000,
              0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0118, 0.0000,
              0.4471, 0.6314, 0.6314, 0.6510, 0.6235, 0.6588, 0.6314, 0.6314,
              0.6745, 0.6353, 0.6471, 0.6706, 0.1961, 0.0000, 0.0196, 0.0000,
              0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0039, 0.0000,
              0.5843, 0.6157, 0.6549, 0.6745, 0.6235, 0.6745, 0.6431, 0.6314,
              0.6745, 0.6667, 0.6275, 0.6706, 0.3490, 0.0000, 0.0157, 0.0000,
              0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0078, 0.0000, 0.0157,
              0.6706, 0.6431, 0.6510, 0.6784, 0.6235, 0.7020, 0.6510, 0.6275,
              0.6824, 0.6549, 0.6353, 0.6510, 0.5020, 0.0000, 0.0078, 0.0000,
              0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0118, 0.0000, 0.0706,
              0.5961, 0.6784, 0.6275, 0.7020, 0.6039, 0.7098, 0.6510, 0.6431,
              0.6863, 0.6667, 0.6510, 0.6667, 0.6431, 0.0000, 0.0000, 0.0039,
              0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0157, 0.0000, 0.1843,
              0.6471, 0.6745, 0.6549, 0.7255, 0.6000, 0.7333, 0.6784, 0.6471,
              0.6824, 0.7020, 0.6510, 0.6510, 0.6196, 0.0196, 0.0000, 0.0118,
              0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0157, 0.0000, 0.3412,
              0.7059, 0.6353, 0.7020, 0.7020, 0.6157, 0.7490, 0.7137, 0.6471,
              0.6588, 0.7451, 0.6784, 0.6471, 0.6510, 0.0784, 0.0000, 0.0157,
              0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0157, 0.0000, 0.4118,
              0.7333, 0.6157, 0.7608, 0.6863, 0.6314, 0.7451, 0.7216, 0.6667,
              0.6196, 0.8039, 0.6941, 0.6588, 0.6706, 0.1725, 0.0000, 0.0157,
              0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0196, 0.0000, 0.5412,
              0.7098, 0.6196, 0.8039, 0.6275, 0.6549, 0.7451, 0.7765, 0.6549,
              0.5961, 0.8549, 0.7294, 0.6667, 0.6745, 0.2235, 0.0000, 0.0196,
              0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0196, 0.0000, 0.5294,
              0.6824, 0.6549, 0.7804, 0.6078, 0.6510, 0.7882, 0.8588, 0.6471,
              0.6196, 0.8549, 0.7373, 0.6549, 0.6863, 0.2196, 0.0000, 0.0275,
              0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0196, 0.0000, 0.5059,
              0.6706, 0.6745, 0.6941, 0.6000, 0.6235, 0.8078, 0.8471, 0.5804,
              0.6157, 0.8078, 0.7451, 0.6471, 0.6863, 0.1882, 0.0000, 0.0196,
              0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0196, 0.0000, 0.6549,
              0.7333, 0.7137, 0.7765, 0.7608, 0.7843, 0.8863, 0.9412, 0.7216,
              0.8078, 1.0000, 0.7725, 0.6980, 0.7020, 0.1647, 0.0000, 0.0196,
              0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0118, 0.0000, 0.4510,
              0.5294, 0.4431, 0.4157, 0.3333, 0.3216, 0.4235, 0.5216, 0.3255,
              0.3529, 0.4745, 0.4706, 0.4314, 0.6196, 0.0706, 0.0000, 0.0118,
              0.0000, 0.0000, 0.0000, 0.0000]]]) 0
    




    Text(0.5, 1.0, 'T-shirt/top')




    
![png](Learn%20Pytorch_files/Learn%20Pytorch_73_2.png)
    



```python
# How many samples are there? 
len(train_data.data), len(train_data.targets), len(test_data.data), len(test_data.targets)
```




    (60000, 60000, 10000, 10000)




```python
# Plot more images
torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False);
```


    
![png](Learn%20Pytorch_files/Learn%20Pytorch_75_0.png)
    



```python
from torch.utils.data import DataLoader

# Setup the batch size hyperparameter
BATCH_SIZE = 32

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(train_data, # dataset to turn into iterable
    batch_size=BATCH_SIZE, # how many samples per batch? 
    shuffle=True # shuffle data every epoch?
)

test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False # don't necessarily have to shuffle the testing data
)

# Let's check out what we've created
print(f"Dataloaders: {train_dataloader, test_dataloader}") 
print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")
```

    Dataloaders: (<torch.utils.data.dataloader.DataLoader object at 0x0000026C54D5D6D0>, <torch.utils.data.dataloader.DataLoader object at 0x0000026C4D350990>)
    Length of train dataloader: 1875 batches of 32
    Length of test dataloader: 313 batches of 32
    


```python
# Check out what's inside the training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
train_features_batch.shape, train_labels_batch.shape
```




    (torch.Size([32, 1, 28, 28]), torch.Size([32]))




```python
# Show a sample
torch.manual_seed(42)
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
plt.imshow(img.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis("Off");
print(f"Image size: {img.shape}")
print(f"Label: {label}, label size: {label.shape}")
```

    Image size: torch.Size([1, 28, 28])
    Label: 6, label size: torch.Size([])
    


    
![png](Learn%20Pytorch_files/Learn%20Pytorch_78_1.png)
    



```python
from torch import nn
class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # neural networks like their inputs in vector form
            nn.Linear(in_features=input_shape, out_features=hidden_units), # in_features = number of features in a data sample (784 pixels)
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    
    def forward(self, x):
        return self.layer_stack(x)
```


```python
torch.manual_seed(42)

# Need to setup model with input parameters
model_6 = FashionMNISTModelV0(input_shape=784, # one for every pixel (28x28)
    hidden_units=10, # how many units in the hiden layer
    output_shape=len(class_names) # one for every class
)
model_6.to(device) # keep model on CPU to begin with 
```




    FashionMNISTModelV0(
      (layer_stack): Sequential(
        (0): Flatten(start_dim=1, end_dim=-1)
        (1): Linear(in_features=784, out_features=10, bias=True)
        (2): ReLU()
        (3): Linear(in_features=10, out_features=10, bias=True)
      )
    )




```python
from timeit import default_timer as timer 
def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time
```


```python
loss_fn_6 = nn.CrossEntropyLoss()
optimizer_6 = torch.optim.Adam(params=model_6.parameters())
```


```python
# Import tqdm for progress bar
from tqdm.auto import tqdm

# Set the seed and start the timer
torch.manual_seed(42)
train_time_start_on_cpu = timer()

# Set the number of epochs (we'll keep this small for faster training times)
epochs = 3

# Create training and testing loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    ### Training
    train_loss = 0
    # Add a loop to loop through training batches
    for batch, (X, y) in enumerate(train_dataloader):
        model_6.train() 
        # 1. Forward pass
        y_pred = model_6(X)
        # 2. Calculate loss (per batch)
        loss = loss_fn_6(y_pred, y)
        train_loss += loss # accumulatively add up the loss per epoch 

        # 3. Optimizer zero grad
        optimizer_6.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer_6.step()

        # Print out how many samples have been seen
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")

    # Divide total train loss by length of train dataloader (average loss per batch per epoch)
    train_loss /= len(train_dataloader)
    
    ### Testing
    # Setup variables for accumulatively adding up loss and accuracy 
    test_loss, test_acc = 0, 0 
    model_6.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            # 1. Forward pass
            test_pred_logit = model_6(X)
           
            # 2. Calculate loss (accumatively)
            test_loss += loss_fn_6(test_pred_logit, y) # accumulatively add up the loss per epoch
            test_pred=torch.softmax(test_pred_logit,dim=1).argmax(dim=1)

            # 3. Calculate accuracy (preds need to be same as y_true)
            test_acc += accuracy1(y_true=y, y_pred=test_pred)
        
        # Calculations on test metrics need to happen inside torch.inference_mode()
        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_dataloader)

        # Divide total accuracy by length of test dataloader (per batch)
        test_acc /= len(test_dataloader)

    ## Print out what's happening
    print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

# Calculate training time      
train_time_end_on_cpu = timer()
total_train_time_model_6 = print_train_time(start=train_time_start_on_cpu, 
                                           end=train_time_end_on_cpu,
                                           device=str(next(model_6.parameters()).device))
```


      0%|          | 0/3 [00:00<?, ?it/s]


    Epoch: 0
    -------
    Looked at 0/60000 samples
    Looked at 12800/60000 samples
    Looked at 25600/60000 samples
    Looked at 38400/60000 samples
    Looked at 51200/60000 samples
    
    Train loss: 0.41809 | Test loss: 0.45716, Test acc: 83.77%
    
    Epoch: 1
    -------
    Looked at 0/60000 samples
    Looked at 12800/60000 samples
    Looked at 25600/60000 samples
    Looked at 38400/60000 samples
    Looked at 51200/60000 samples
    
    Train loss: 0.40803 | Test loss: 0.44582, Test acc: 84.11%
    
    Epoch: 2
    -------
    Looked at 0/60000 samples
    Looked at 12800/60000 samples
    Looked at 25600/60000 samples
    Looked at 38400/60000 samples
    Looked at 51200/60000 samples
    
    Train loss: 0.39891 | Test loss: 0.44145, Test acc: 84.29%
    
    Train time on cpu: 53.915 seconds
    


```python
# Move values to device
torch.manual_seed(42)
def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn, 
               device: torch.device = device):
    """Evaluates a given model on a given dataset.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.
        device (str, optional): Target device to compute on. Defaults to device.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to the target device
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        
        # Scale loss and acc
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}

# Calculate model 1 results with device-agnostic code 
model_6_results = eval_model(model=model_6, data_loader=test_dataloader,
    loss_fn=loss_fn_6, accuracy_fn=accuracy1,
    device=device
)
model_6_results
```




    {'model_name': 'FashionMNISTModelV0',
     'model_loss': 0.4414539635181427,
     'model_acc': 84.28514376996804}




```python
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=torch.softmax(test_pred,dim=1).argmax(dim=1) # Go from logits -> pred labels
            )
        
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
```


```python
torch.manual_seed(42)

# Measure time
from timeit import default_timer as timer
train_time_start_on_gpu = timer()

epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader, 
        model=model_6, 
        loss_fn=loss_fn_6,
        optimizer=optimizer_6,
        accuracy_fn=accuracy1
    )
    test_step(data_loader=test_dataloader,
        model=model_6,
        loss_fn=loss_fn_6,
        accuracy_fn=accuracy1
    )

train_time_end_on_gpu = timer()
total_train_time_model_0 = print_train_time(start=train_time_start_on_gpu,
                                            end=train_time_end_on_gpu,
                                            device=device)
```


      0%|          | 0/3 [00:00<?, ?it/s]


    Epoch: 0
    ---------
    Train loss: 0.37740 | Train accuracy: 86.84%
    Test loss: 0.43533 | Test accuracy: 84.55%
    
    Epoch: 1
    ---------
    Train loss: 0.37402 | Train accuracy: 86.81%
    Test loss: 0.42221 | Test accuracy: 85.22%
    
    Epoch: 2
    ---------
    Train loss: 0.36964 | Train accuracy: 87.06%
    Test loss: 0.42417 | Test accuracy: 85.09%
    
    Train time on cpu: 24.191 seconds
    


```python
# Create a convolutional neural network 
class FashionMNISTModelV2(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units*7*7, 
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

torch.manual_seed(42)
model_7 = FashionMNISTModelV2(input_shape=1, 
    hidden_units=10, 
    output_shape=len(class_names)).to(device)
model_7
```




    FashionMNISTModelV2(
      (block_1): Sequential(
        (0): Conv2d(1, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU()
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (block_2): Sequential(
        (0): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU()
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (classifier): Sequential(
        (0): Flatten(start_dim=1, end_dim=-1)
        (1): Linear(in_features=490, out_features=10, bias=True)
      )
    )




```python
loss_fn_7 = nn.CrossEntropyLoss()
optimizer_7 = torch.optim.Adam(params=model_7.parameters())
```


```python
torch.manual_seed(42)

# Measure time
from timeit import default_timer as timer
train_time_start_on_gpu = timer()

epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader, 
        model=model_7, 
        loss_fn=loss_fn_7,
        optimizer=optimizer_7,
        accuracy_fn=accuracy1
    )
    test_step(data_loader=test_dataloader,
        model=model_7,
        loss_fn=loss_fn_7,
        accuracy_fn=accuracy1
    )

train_time_end_on_gpu = timer()
total_train_time_model_1 = print_train_time(start=train_time_start_on_gpu,
                                            end=train_time_end_on_gpu,
                                            device=device)
```


      0%|          | 0/3 [00:00<?, ?it/s]


    Epoch: 0
    ---------
    Train loss: 0.53521 | Train accuracy: 80.52%
    Test loss: 0.39757 | Test accuracy: 85.79%
    
    Epoch: 1
    ---------
    Train loss: 0.36155 | Train accuracy: 86.93%
    Test loss: 0.38134 | Test accuracy: 86.16%
    
    Epoch: 2
    ---------
    Train loss: 0.32650 | Train accuracy: 88.33%
    Test loss: 0.33391 | Test accuracy: 87.86%
    
    Train time on cpu: 65.576 seconds
    


```python
# Get model_2 results 
model_7_results = eval_model(
    model=model_7,
    data_loader=test_dataloader,
    loss_fn=loss_fn_7,
    accuracy_fn=accuracy1
)
model_7_results
```




    {'model_name': 'FashionMNISTModelV2',
     'model_loss': 0.33390533924102783,
     'model_acc': 87.8594249201278}




```python
import pandas as pd
compare_results = pd.DataFrame([model_6_results, model_7_results])
compare_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model_name</th>
      <th>model_loss</th>
      <th>model_acc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FashionMNISTModelV0</td>
      <td>0.441454</td>
      <td>84.285144</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FashionMNISTModelV2</td>
      <td>0.333905</td>
      <td>87.859425</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Add training times to results comparison
compare_results["training_time"] = [total_train_time_model_0,
                                    total_train_time_model_1]
compare_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model_name</th>
      <th>model_loss</th>
      <th>model_acc</th>
      <th>training_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FashionMNISTModelV0</td>
      <td>0.441454</td>
      <td>84.285144</td>
      <td>24.190750</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FashionMNISTModelV2</td>
      <td>0.333905</td>
      <td>87.859425</td>
      <td>65.575683</td>
    </tr>
  </tbody>
</table>
</div>




```python
def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare sample
            sample = torch.unsqueeze(sample, dim=0).to(device) # Add an extra dimension and send sample to device

            # Forward pass (model outputs raw logit)
            pred_logit = model(sample)

            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 1, so can perform on dim=0)

            # Get pred_prob off GPU for further calculations
            pred_probs.append(pred_prob.cpu())
            
    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)
```


```python
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# 2. Setup confusion matrix instance and compare predictions to targets
confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_preds,
                         target=test_data.targets)

# 3. Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy 
    class_names=class_names, # turn the row and column labels into class names
    figsize=(10, 7)
)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[263], line 6
          4 # 2. Setup confusion matrix instance and compare predictions to targets
          5 confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
    ----> 6 confmat_tensor = confmat(preds=y_preds,
          7                          target=test_data.targets)
          9 # 3. Plot the confusion matrix
         10 fig, ax = plot_confusion_matrix(
         11     conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy 
         12     class_names=class_names, # turn the row and column labels into class names
         13     figsize=(10, 7)
         14 )
    

    File ~\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\nn\modules\module.py:1518, in Module._wrapped_call_impl(self, *args, **kwargs)
       1516     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1517 else:
    -> 1518     return self._call_impl(*args, **kwargs)
    

    File ~\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\nn\modules\module.py:1527, in Module._call_impl(self, *args, **kwargs)
       1522 # If we don't have any hooks, we want to skip the rest of the logic in
       1523 # this function, and just call forward.
       1524 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
       1525         or _global_backward_pre_hooks or _global_backward_hooks
       1526         or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1527     return forward_call(*args, **kwargs)
       1529 try:
       1530     result = None
    

    File ~\AppData\Local\Programs\Python\Python311\Lib\site-packages\torchmetrics\metric.py:312, in Metric.forward(self, *args, **kwargs)
        310     self._forward_cache = self._forward_full_state_update(*args, **kwargs)
        311 else:
    --> 312     self._forward_cache = self._forward_reduce_state_update(*args, **kwargs)
        314 return self._forward_cache
    

    File ~\AppData\Local\Programs\Python\Python311\Lib\site-packages\torchmetrics\metric.py:381, in Metric._forward_reduce_state_update(self, *args, **kwargs)
        378 self._enable_grad = True  # allow grads for batch computation
        380 # calculate batch state and compute batch value
    --> 381 self.update(*args, **kwargs)
        382 batch_val = self.compute()
        384 # reduce batch and global state
    

    File ~\AppData\Local\Programs\Python\Python311\Lib\site-packages\torchmetrics\metric.py:483, in Metric._wrap_update.<locals>.wrapped_func(*args, **kwargs)
        481 with torch.set_grad_enabled(self._enable_grad):
        482     try:
    --> 483         update(*args, **kwargs)
        484     except RuntimeError as err:
        485         if "Expected all tensors to be on" in str(err):
    

    File ~\AppData\Local\Programs\Python\Python311\Lib\site-packages\torchmetrics\classification\confusion_matrix.py:283, in MulticlassConfusionMatrix.update(self, preds, target)
        281 """Update state with predictions and targets."""
        282 if self.validate_args:
    --> 283     _multiclass_confusion_matrix_tensor_validation(preds, target, self.num_classes, self.ignore_index)
        284 preds, target = _multiclass_confusion_matrix_format(preds, target, self.ignore_index)
        285 confmat = _multiclass_confusion_matrix_update(preds, target, self.num_classes)
    

    File ~\AppData\Local\Programs\Python\Python311\Lib\site-packages\torchmetrics\functional\classification\confusion_matrix.py:267, in _multiclass_confusion_matrix_tensor_validation(preds, target, num_classes, ignore_index)
        265     raise ValueError("If `preds` have one dimension more than `target`, `preds` should be a float tensor.")
        266 if preds.shape[1] != num_classes:
    --> 267     raise ValueError(
        268         "If `preds` have one dimension more than `target`, `preds.shape[1]` should be"
        269         " equal to number of classes."
        270     )
        271 if preds.shape[2:] != target.shape[1:]:
        272     raise ValueError(
        273         "If `preds` have one dimension more than `target`, the shape of `preds` should be"
        274         " (N, C, ...), and the shape of `target` should be (N, ...)."
        275     )
    

    ValueError: If `preds` have one dimension more than `target`, `preds.shape[1]` should be equal to number of classes.



```python
from pathlib import Path

# Create models directory (if it doesn't already exist), see: https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                 exist_ok=True # if models directory already exists, don't error
)

# Create model save path
MODEL_NAME = "03_pytorch_computer_vision_model_2.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_7.state_dict(), # only saving the state_dict() only saves the learned parameters
           f=MODEL_SAVE_PATH)
```

    Saving model to: models\03_pytorch_computer_vision_model_2.pth
    


```python

```
