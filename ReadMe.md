<p align="center"><a href="https://www.torontomu.ca/akhademi/">
  <img src="Images/IAMLab-Logo.jpg" height=120>
</a></p>


<!-- omit in toc -->
# [Reliability of Foundation Models for Image Retrieval in Histopathology]()
<!-- omit in toc -->

## Cross-Scanner Image Retrieval Reliability Evaluation Framework

a) Patch-Level Retrieval
<p align="center">
<img width="800" src="./Images/PatchLevel_Retrieval.jpg"> 
</p>
b) TMA-Level Retrieval
<p align="center">
<img width="800" src="./Images/TMALevel_Retrieval.jpg"> 
</p>

Ensuring fairness and explainability is essential for the development of ethical, reliable, and effective AI systems in healthcare. Bias in AI models can contribute to disparities in clinical outcomes, challenging equity in medical decision-making. Content-Based Image Retrieval (CBIR) offers interpretable, visual tools to support diagnostic processes; however, these tools remain susceptible to biases inherent in the data. This study examines fairness and explainability in AI systems for healthcare, focusing on bias in CBIR for histopathology. Specifically, it investigates how differences between scanning devices can introduce covariate bias into Foundation Models (FMs). To enable this analysis, the authors created a unique dataset of spatially aligned histopathology images scanned by two different devices, allowing them to directly study the impact of scanner variability on FM representations.

## Patch-Level Retrieval
**Given that all the Embeddings of Spatially corresponding ID and OOD patches are already extracted and stored with corresponding ID and OOD indices:**
<br>
Patch_Level_Search Notebook [Patch_Level_Search.ipynb](https://github.com/IAMLAB-Ryerson/Reliability_Search-Retrieval/blob/main/Patch_Level_Search.ipynb) takes in already generated embeddings, labels and image paths from the corresponding registered patches for ID and OOD **Query** and **Archive** in order to evaluate the cross-scanner retrieval reliability.

<br>
```
Expected input format:<br>
**ID and OOD Query:** <br>
── ID-All_Test_Embeds / Labels / file_paths               | ── OOD-All_Test_Embeds / Labels / file_paths
    ├── ID Test Patch / Label / file_path 1               |    ├── ID Test Patch / Label / file_path 1
    ├── ID Test Patch / Label / file_path 2               |    ├── ID Test Patch / Label / file_path 2
    ├── ID Test Patch / Label / file_path 3               |    ├── ID Test Patch / Label / file_path 3
    .                                                     |    .
    .                                                     |    .
    .                                                     |    .

<br>
```

First, ID Query patches are matched againt the ID Archive patches to find similar patches using Euclidean, cosine, and Hammind distances.
Second, for the cross-scanner retrieval, OOD Query Patches are matched against the ID Archive to find the most similar patches using Euclidean, cosine, and Hammind distances.

<br>

All the results are saved in the "current working directory".

## Slide-Level Retrieval
**Given that all the Embeddings of Spatially corresponding ID and OOD patches are already extracted and stored with corresponding ID and OOD indices:**

## Citation
```
@article{shafique2026reliability,
  title={Reliability of Foundation Models for Image Retrieval in Histopathology},
  author={Shafique, Abubakr and Qin, Xiaoli and Dy, Amanda and Alshamlan, Najd and Androutsos, Dimitrios and Done, Susan J and Khademi, April},
  journal={npj Imaging},
  volume={},
  number={},
  pages={},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```

## Disclaimer
This code is intended for research purposes only. Any commercial use is prohibited.




