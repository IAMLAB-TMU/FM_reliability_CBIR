<p align="center"><a href="https://www.torontomu.ca/akhademi/">
  <img src="Images/IAMLab-Logo.jpg" height=120>
</a></p>


<!-- omit in toc -->
# [Reliability of Foundation Models for Image Retrieval in Histopathology]()
<!-- omit in toc -->

## Cross-Scanner Reliability Evaluation Framework

a) Patch-Level Retrieval
<p align="center">
<img width="800" src="./Images/PatchLevel_Retrieval.jpg"> 
</p>
b) TMA-Level Retrieval
<p align="center">
<img width="800" src="./Images/TMALevel_Retrieval.jpg"> 
</p>

Domain Shift is a remaining challenge in histopathology image analysis and limits the reliable application in real-world clinical settings. Developing domain-generalized models is critical for ensuring robust generalization across datasets from different domains. For this, we proposed HistoLite: a lightweight, self-supervised domain-generalized representation learning framework that is both resource-efficient and customizable. HistoLite is designed to be trained on a personal computer equipped with a decent GPU, making it accessible for researchers and institutions with limited resources.

## HistoLite Encoder Inference on Histopathology Image
**To extract embeddings from histopathology images using the pretrained HistoLite encoder:**

First, download the pretrained model ```HistoLite_Encoder_512_14epochs.pth``` from the [GitHub HistoLite repo](https://github.com/IAMLAB-Ryerson/HistoLite/tree/main/Weights). Then, locate it in the ```./inference``` directory.

```python
from HistoLite_Inference import get_HistoLite_Encoder
from PIL import Image
import torch

histoImg = Image.open('./inference/img.png')
model, transformInput = get_HistoLite_Encoder(weights_path='./inference/HistoLite_Encoder_512_14epochs.pth', Inputsize=512)

img = transformInput(histoImg)
embedding = model(img.unsqueeze(0))

print(embedding.shape)

```


## HistoLite SSL Training on Histopathology Images
**To train your own model using HistoLite SSL Framework with custom dataset:**

To train the model:

```python
from HistoLite_SSL_Framework import get_HistoLite_SSL
import torch
from torchsummary import summary

HistoLite_SSL = get_HistoLite_SSL()
HistoLite_SSL = HistoLite_SSL.to("cuda")

summary(HistoLite_SSL, [(3, 512, 512), (3, 512, 512)])


## In order to Train:
## Loss Function
criterion_MSE = nn.MSELoss()

# Forward pass
Feat1, DE1, Feat2, DE2, P1, P2 = HistoLite_SSL(Orig_Input, Aug_Input)

loss_Orig = criterion_MSE(DE1, Orig_Input)
loss_Aug = criterion_MSE(DE2, Aug_Input)
loss_feat_sim = (criterion_MSE(P1, Feat2) + criterion_MSE(P2, Feat1)) * 0.5

## Combined Loss is used for Backward pass and optimization
loss = loss_Orig + loss_Aug + loss_feat_sim

# Backward pass and optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()

```

For complete training pipline, sample trining code is provided here: ```Train_HistoLite_Sample.ipynb``` from the [GitHub HistoLite repo](https://github.com/IAMLAB-Ryerson/HistoLite/blob/main/Train_HistoLite_Sample.ipynb).


## Citation
```
@article{shafique2025lightweight,
  title={Lightweight self supervised learning framework for domain generalization in histopathology},
  author={Shafique, Abubakr and Dy, Amanda and Qin, Xiaoli and Alshamlan, Najd and Androutsos, Dimitrios and Done, Susan J and Khademi, April},
  journal={Scientific Reports},
  volume={15},
  number={1},
  pages={36631},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

## Disclaimer
This code is intended for research purposes only. Any commercial use is prohibited.




