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
```Patch_Level_Search``` Notebook [Patch_Level_Search.ipynb](https://github.com/IAMLAB-Ryerson/Reliability_Search-Retrieval/blob/main/Patch_Level_Search.ipynb) takes in already generated embeddings, labels and image paths from the corresponding registered patches for ID and OOD **Query** and **Archive** in order to evaluate the cross-scanner retrieval reliability.
<br>
**Expected patch-level input format:**
```
ScannerA Query:                                             | ScannerB Query: 
── ScannerA-All_Query_Embeds / Labels / file_paths          | ── ScannerB-All_Query_Embeds / Labels / file_paths
    ├── ScannerA Query Patch / Label / patch_path 1         |    ├── ScannerB Query Patch / Label / patch_path 1
    ├── ScannerA Query Patch / Label / patch_path 2         |    ├── ScannerB Query Patch / Label / patch_path 2
    ├── ScannerA Query Patch / Label / patch_path 3         |    ├── ScannerB Query Patch / Label / patch_path 3
    .                                                       |    .
    .                                                       |    .
    .                                                       |    .
                                                            |
ScannerA Archive:                                           | ScannerB Archive: 
── ScannerA-All_Archive_Embeds / Labels / file_paths        | ── ScannerB-All_Archive_Embeds / Labels / file_paths
    ├── ScannerA Archive Patch / Label / patch_path 1       |    ├── ScannerB Archive Patch / Label / patch_path 1
    ├── ScannerA Archive Patch / Label / patch_path 2       |    ├── ScannerB Archive Patch / Label / patch_path 2
    ├── ScannerA Archive Patch / Label / patch_path 3       |    ├── ScannerB Archive Patch / Label / patch_path 3
    .                                                       |    .
    .                                                       |    .
    .                                                       |    .
```
<br>
First, ID Query patches are matched againt the ID Archive patches to find similar patches using Euclidean, cosine, and Hammind distances.
Second, for the cross-scanner retrieval, OOD Query Patches are matched against the ID Archive to find the most similar patches using Euclidean, cosine, and Hammind distances.

<br>
All the results are saved in the "current working directory".

## Slide-Level Retrieval
**Given that all the Embeddings of Spatially corresponding ID and OOD patches are already extracted and stored with corresponding ID and OOD indices:**
<br>
```TMA_Level_Search``` Notebook [TMA_Level_Search.ipynb](https://github.com/IAMLAB-Ryerson/Reliability_Search-Retrieval/blob/main/TMA_Level_Search.ipynb) takes in already generated TMA embeddings, labels and image paths from the corresponding registered slides from two scanners in order to evaluate the cross-scanner retrieval reliability.
<br>
**Expected slide-level input format:**
```
ScannerA Slides:
── ScannerA-All_Query_Embeds / Labels / file_paths
    ├── ScannerA Query Slide / Label / slide_path 1
    |   |── patch 1 Embeds
    |   |── patch 2 Embeds
    |   |── patch 3 Embeds
    |   ...
    ├── ScannerA Query Slide / Label / slide_path 2
    |   |── patch 1 Embeds
    |   |── patch 2 Embeds
    |   |── patch 3 Embeds
    |   ...
    ├── ScannerA Query Slide / Label / slide_path 3
    |   |── patch 1 Embeds
    |   |── patch 2 Embeds
    |   |── patch 3 Embeds
    |   ....  
    .
    .
    .
ScannerB Slides:
── ScannerB-All_Archive_Embeds / Labels / file_paths
    ├── ScannerB Archive Slide / Label / slide_path 1
    |   |── patch 1 Embeds
    |   |── patch 2 Embeds
    |   |── patch 3 Embeds
    |   ....
    ├── ScannerB Archive Slide / Label / slide_path 2
    |   |── patch 1 Embeds
    |   |── patch 2 Embeds
    |   |── patch 3 Embeds
    |   ....
    ├── ScannerB Archive Slide / Label / slide_path 3
    |   |── patch 1 Embeds
    |   |── patch 2 Embeds
    |   |── patch 3 Embeds
    |   ....
    .
    .
    .
```
<br>

For TMA/slide-level evaluation, a leave-one-patient-out cross-validation strategy was employed to ensure comprehensive assessment across the entire dataset. In the TMA setting, all available patches were utilized for retrieval and matching, with similarity quantified using the median of the minimum pairwise distances.<br>

```Optional:``` For Whole Slides, Yottixel mosaic [Generate_Mosaic.ipynb](https://github.com/IAMLAB-Ryerson/Reliability_Search-Retrieval/blob/main/Generate_Mosaic.ipynb) can be used to get coordinates for the selected patches, and the coordinates are used to get embeddings from the registered slides from both scanners using [Generate_Mosaic_Embeds.ipynb](https://github.com/IAMLAB-Ryerson/Reliability_Search-Retrieval/blob/main/Generate_Mosaic_Embeds.ipynb).<br>

All the results are saved in the "current working directory".<br>

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




