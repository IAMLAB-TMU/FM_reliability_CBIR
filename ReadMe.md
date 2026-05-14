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
```Patch_Level_Search``` Python script [Patch_Level_Search.py](https://github.com/IAMLAB-Ryerson/FM_reliability_CBIR/blob/main/Patch_Level_Search.py) takes in already generated embeddings, labels and image paths from the corresponding registered patches for ID and OOD **Query** and **Archive** in order to evaluate the cross-scanner retrieval reliability.
<br>

```
Usage: python Patch_Level_Search.py [options]...

--seed                      Set the seed number
--folds                     Number of folds of the available data
--ID_Query_Embeds_dir       Set the data path for ID Query data embeddings
--ID_Query_Labels_dir       Set the data path for ID Query data Labels
--ID_Query_Names_dir        Set the data path for ID Query data path list
--OOD_Query_Embeds_dir      Set the data path for OOD Query data embeddings
--OOD_Query_Labels_dir      Set the data path for OOD Query data Labels
--OOD_Query_Names_dir       Set the data path for OOD Query data path list
--ID_Archive_Embeds_dir     Set the data path for ID Archive data embeddings
--ID_Archive_Labels_dir     Set the data path for ID Archive data Labels
--ID_Archive_Names_dir      Set the data path for ID Archive data path list
--output_dir                Set the output dir
--model_name                choose the model, e.g., UNI
--ID_Data                   Select which one is the ID data, e.g., Scanner A
--OOD_Data                  Select which one is the OOD data, e.g., Scanner B

```
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
All the results are saved in the "current working directory". <br>

## Slide-Level Retrieval
**Given that all the Embeddings of Spatially corresponding ID and OOD patches are already extracted and stored with corresponding ID and OOD indices:** <br>

```TMA_Level_Search``` Python script [TMA_Level_Search.py](https://github.com/IAMLAB-Ryerson/FM_reliability_CBIR/blob/main/TMA_Level_Search.py) takes in already generated TMA embeddings, labels and image paths from the corresponding registered slides from two scanners in order to evaluate the cross-scanner retrieval reliability.
<br>

```
Usage: python TMA_Level_Search.py [options]...

--seed                      Set the seed number
--ID_Data_dir               Set the data path for ID data embeddings
--OOD_Data_dir              Set the data path for OOD data embeddings
--csv_file_dir              Set the data path for csv file with patient information
--output_dir                Set the output dir
--model_name                choose the model, e.g., UNI
--ID_Data                   Select which one is the ID data, e.g., Scanner A
--OOD_Data                  Select which one is the OOD data, e.g., Scanner B
--TMA_ScannerA_Thumb_dir    Set the data path for Scanner A slide thumbnails.
--TMA_ScannerB_Thumb_dir    Set the data path for Scanner B slide thumbnails.

```
<br>

**Expected slide-level input format:**
```
ScannerA Slides:
── ScannerA-All_Slides_Embeds / Labels / file_paths
    ├── ScannerA Slide / Label / slide_path 1
    |   |── patch 1 Embeds
    |   |── patch 2 Embeds
    |   |── patch 3 Embeds
    |   ...
    ├── ScannerA Slide / Label / slide_path 2
    |   |── patch 1 Embeds
    |   |── patch 2 Embeds
    |   |── patch 3 Embeds
    |   ...
    ├── ScannerA Slide / Label / slide_path 3
    |   |── patch 1 Embeds
    |   |── patch 2 Embeds
    |   |── patch 3 Embeds
    |   ....  
    .
    .
    .
ScannerB Slides:
── ScannerB-All_Slides_Embeds / Labels / file_paths
    ├── ScannerB Slide / Label / slide_path 1
    |   |── patch 1 Embeds
    |   |── patch 2 Embeds
    |   |── patch 3 Embeds
    |   ....
    ├── ScannerB Slide / Label / slide_path 2
    |   |── patch 1 Embeds
    |   |── patch 2 Embeds
    |   |── patch 3 Embeds
    |   ....
    ├── ScannerB Slide / Label / slide_path 3
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

```Optional:``` For whole-slide analysis, the Yottixel mosaic approach could be employed. The [Generate_Mosaic.ipynb](https://github.com/IAMLAB-Ryerson/Reliability_Search-Retrieval/blob/main/Generate_Mosaic.ipynb) notebook was used to obtain coordinates for the selected patches, which were subsequently utilized by [Generate_Mosaic_Embeds.ipynb](https://github.com/IAMLAB-Ryerson/Reliability_Search-Retrieval/blob/main/Generate_Mosaic_Embeds.ipynb) to extract embeddings from the spatially registered slides acquired from both scanners.<br>

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




