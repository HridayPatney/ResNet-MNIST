# 🧠 ResNet on MNIST

This project implements a **Residual Neural Network (ResNet)** from scratch in **PyTorch** and trains it on the **MNIST handwritten digit dataset**.  
ResNets, introduced in *He et al. (2015)*, use **skip connections** to make deep neural networks easier to optimize, solving the vanishing gradient problem.

Instead of learning a direct mapping, ResNet blocks learn a **residual function**:

$$
y = F(x) + x
$$

where \(F(x)\) is the learned transformation and \(x\) is the identity mapping (skip connection).

---

## 🚀 Features
- ✅ Custom ResNet implementation in PyTorch (no pre-built models used)  
- ✅ MNIST dataset loading with `torchvision` (auto-downloads)  
- ✅ GPU/CPU compatible for Google Colab or local training  
- ✅ Training & evaluation loops with progress + accuracy  
- ✅ Adaptive Average Pooling for handling small 28×28 inputs  

---

## 📊 Dataset
- **MNIST**: 70,000 handwritten digits (60,000 training + 10,000 testing)  
- Images are grayscale, **28×28 pixels**  
- **10 classes**: digits `0` through `9`

---

## ⚙️ Model Architecture
- Initial **Conv + BatchNorm + ReLU**  
- **3 ResNet stages**:
  - Stage 1: 16 channels  
  - Stage 2: 32 channels (downsampling)  
  - Stage 3: 64 channels (downsampling)  
- **Adaptive Average Pooling** → Flatten → Fully Connected Layer → 10 outputs  

---

## 📈 Results
After 5–10 epochs of training on GPU (Colab recommended):  
- Training loss decreases steadily  
- Test accuracy reaches **~99%**  

---

## 🔧 How to Run

### 1. Clone Repository
```bash
git clone https://github.com/your-username/resnet-mnist.git
cd resnet-mnist
```
### 2. Install Dependencies
```bash
pip install torch torchvision
```
### 3. Run Training
```bash
python train.py

```
