# RDiffBR: Modeling Item-Level Dynamic Variability with Residual Diffusion for Bundle Recommendation(AAAI'26)
<img width="1452" height="634" alt="image" src="https://github.com/user-attachments/assets/8ed73876-2b37-45de-b41e-226514910a6e" />

## 🛠 Requirements
```
PyTorch  1.8.1
Python  3.8
CUDA  11.1
```

## 📁 Dataset preprocess
Preprocess the datasets' bundle_item.txt using 
```
./Fold_Mask_bundle_dataset.py
```
## 🚀 Running Experiments

### Step 1: Train the Base Model with RDiffBR
```
python train.py 
```

### Step 2: Test in bundle-item dynamic variability
Remove the comments in train.py:
```
model.load_state_dict(torch.load('./model'), strict=False)  #diffusion model
bundle_reverse_model.load_state_dict(torch.load('./ED_1_Neg_1_2048_0.001_0.0001_64_0.2_0.2_0.2_1_0.04_0.25')) #bundle recommendation model
for param in model.parameters():
    param.requires_grad = False  
for param in bundle_reverse_model.parameters():
    param.requires_grad = False  
```
Choose ρ in utility.py:
```
    def get_bi_mask(self):    # mask0.9 -> ρ = -4, mask0.8 -> ρ = -3 ,..., mask0.5 -> ρ = 0,..., mask0.1 -> ρ = 4 , bundle_item.txt -> ρ = 5
        with open(os.path.join(self.path, self.name, 'Fold_bundle_item_mask0.9.txt'), 'r') as f:
```


