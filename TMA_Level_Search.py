import os, sys
import pandas as pd
import numpy as np
import argparse
import random
from sklearn.metrics import classification_report


parser = argparse.ArgumentParser(description='TMA Level Search and Retrieval')

parser.add_argument('--seed', type=int, default=8)

parser.add_argument('--ID_Data_dir', type=str, default="Data Dir for ID TMA Embeds")
parser.add_argument('--OOD_Data_dir', type=str, default="Data Dir for OOD TMA Embeds")
parser.add_argument('--csv_file_dir', type=str, default="CSV file path to get patient information")
parser.add_argument('--output_dir', type=str, default=os.getcwd())
parser.add_argument('--model_name', type=str, default='UNI', choices=['UNI', 'Virchow2', 'GigaPath'])

parser.add_argument('--ID_Data', type=str, default="ScannerA", choices=['ScannerA', 'ScannerB'])
parser.add_argument('--OOD_Data', type=str, default="ScannerB", choices=['ScannerA', 'ScannerB'])

parser.add_argument('--TMA_ScannerA_Thumb_dir', type=str, default="Path to the Dir of ScannerA slide Thumbnails")
parser.add_argument('--TMA_ScannerB_Thumb_dir', type=str, default="Path to the Dir of ScannerB slide Thumbnails")


args = parser.parse_args()

current_working_directory = args.output_dir

### Scanner Type
ID_Data = args.ID_Data
OOD_Data = args.OOD_Data

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

from collections import Counter
def Majority_Vote(Input_Array):

    # Count occurrences
    counter = Counter(Input_Array)

    # Find the majority string
    majority_Label = counter.most_common(1)[0][0]
    return majority_Label


def med_min_search(archive_features, true_labels, WSI_List, Case_List):
    import bitarray
    from bitarray import util as butil
    import math

    top_k = 5

    #convert patch feature vectors into barcodes
    barcodes = []
    for WSI_feat in archive_features:
        patch_barcodes = (np.diff(np.array(WSI_feat), axis=1) < 0)*1
        patch_barcodes = [bitarray.bitarray(b.tolist()) for b in patch_barcodes]
        barcodes.append(patch_barcodes)

    #--------------------------------------------------
    top1_labels = []
    top3_labels = []
    top5_labels = []

    topk_labels = []
    topk_med_min_distances = []
    topk_names = []
    #find the median of the min distances
    for i, _ in enumerate(barcodes):
        WSI_i = barcodes[i]
        med_min = []
        for j, _ in enumerate(barcodes):
            WSI_j = barcodes[j]
            min_dist = []
            if Case_List[i] == Case_List[j]:
                min_dist.append(np.inf)
            else:
                for patch_i in WSI_i:
                    #calculate the distance between every image (patch or WSI) barcode with all other barcodes
                    #compare the calculated distances between barcodes by applying XOR
                    distances = [butil.count_xor(patch_i, patch_j) for patch_j in WSI_j]
                    distances = np.array(distances, dtype=np.float32)

                    #store the min distances. Every patch in WSI i has a distance (min) in min_feat
                    min_dist.append(np.min(distances))

            #find the indext of the median distance among the min distances of the patches belong to WSI i
            med_min.append(np.median(min_dist))

        med_min = np.array(med_min)

        sorted_distances_idx = np.argsort(med_min)        
        topk_labels.append(true_labels[sorted_distances_idx[0:top_k]])

        topk_med_min_distances.append(med_min[sorted_distances_idx[0:top_k]])
        topk_names.append(WSI_List[sorted_distances_idx[0:top_k]])

        top1_labels.append(true_labels[sorted_distances_idx[0]])
        top3_labels.append(Majority_Vote(true_labels[sorted_distances_idx[0:3]]))
        top5_labels.append(Majority_Vote(true_labels[sorted_distances_idx[0:5]]))

    topk_labels = np.array(topk_labels)
    topk_med_min_distances = np.array(topk_med_min_distances)
    topk_names = np.array(topk_names)

    #--------------------------------------------------

    return top1_labels, top3_labels, top5_labels, topk_med_min_distances, topk_names, topk_labels


def OOD_med_min_search(ID_archive_features, OOD_archive_features, true_labels, WSI_List, Case_List):
    import bitarray
    from bitarray import util as butil
    import math

    top_k = 5

    #convert patch feature vectors into barcodes
    ID_barcodes = []
    for WSI_feat in ID_archive_features:
        patch_barcodes = (np.diff(np.array(WSI_feat), axis=1) < 0)*1
        patch_barcodes = [bitarray.bitarray(b.tolist()) for b in patch_barcodes]
        ID_barcodes.append(patch_barcodes)

    #convert patch feature vectors into barcodes
    OOD_barcodes = []
    for WSI_feat in OOD_archive_features:
        patch_barcodes = (np.diff(np.array(WSI_feat), axis=1) < 0)*1
        patch_barcodes = [bitarray.bitarray(b.tolist()) for b in patch_barcodes]
        OOD_barcodes.append(patch_barcodes)

    #--------------------------------------------------
    top1_labels = []
    top3_labels = []
    top5_labels = []

    topk_labels = []
    topk_med_min_distances = []
    topk_names = []
    #find the median of the min distances
    for i, _ in enumerate(OOD_barcodes):
        WSI_i = OOD_barcodes[i]
        med_min = []
        for j, _ in enumerate(ID_barcodes):
            WSI_j = ID_barcodes[j]
            min_dist = []
            if Case_List[i] == Case_List[j]:
                min_dist.append(np.inf)
            else:
                for patch_i in WSI_i:
                    #calculate the distance between every image (patch or WSI) barcode with all other barcodes
                    #compare the calculated distances between barcodes by applying XOR
                    distances = [butil.count_xor(patch_i, patch_j) for patch_j in WSI_j]
                    distances = np.array(distances, dtype=np.float32)

                    #store the min distances. Every patch in WSI i has a distance (min) in min_feat
                    min_dist.append(np.min(distances))

            #find the indext of the median distance among the min distances of the patches belong to WSI i
            med_min.append(np.median(min_dist))

        med_min = np.array(med_min)

        sorted_distances_idx = np.argsort(med_min)        
        topk_labels.append(true_labels[sorted_distances_idx[0:top_k]])

        topk_med_min_distances.append(med_min[sorted_distances_idx[0:top_k]])
        topk_names.append(WSI_List[sorted_distances_idx[0:top_k]])

        top1_labels.append(true_labels[sorted_distances_idx[0]])
        top3_labels.append(Majority_Vote(true_labels[sorted_distances_idx[0:3]]))
        top5_labels.append(Majority_Vote(true_labels[sorted_distances_idx[0:5]]))

    topk_labels = np.array(topk_labels)
    topk_med_min_distances = np.array(topk_med_min_distances)
    topk_names = np.array(topk_names)

    #--------------------------------------------------

    return top1_labels, top3_labels, top5_labels, topk_med_min_distances, topk_names, topk_labels


def get_case_number(map_no):
    map_no = int(map_no)
    csv_data = pd.read_csv(csv_file_path)

    Cased = csv_data["Case"]
    Map_Nos = csv_data["Map No"]
    Map_Nos = Map_Nos.to_list()

    return Cased[Map_Nos.index(map_no)]

label_dict = {"T": "Tumour",
              "N": "Normal" }


import PIL
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
import seaborn as sns

# Function to draw a colored border around an image
def add_border(ax, color='black', linewidth=6):
    rect = patches.Rectangle((0, 0), 1, 1, linewidth=linewidth, edgecolor=color, facecolor='none', transform=ax.transAxes)
    ax.add_patch(rect)

def draw_top_k_WSIs(WSI_List, topk_names, topk_med_min_distances, top5_labels, OOD_topk_names, OOD_topk_med_min_distances, OOD_top5_labels, WSI_Labels, idx, Save_Path=None):

    if ID_Data == "ScannerA":
        Q_Img = PIL.Image.open(os.path.join(args.TMA_ScannerA_Thumb_dir, WSI_List[idx][:-11]+".jpg"))  ## Thumbnails for ScannerA thumbnails (ID Query Image)
        OOD_Q_Img = PIL.Image.open(os.path.join(args.TMA_ScannerB_Thumb_dir, WSI_List[idx][:-11]+".jpg")) ## Thumbnails for ScannerB thumbnails (OOD Query Image)

        ID_Imgs = []
        for P in topk_names[idx]:
            ID_Imgs.append(PIL.Image.open(os.path.join(args.TMA_ScannerA_Thumb_dir, P[:-11]+".jpg"))) ## Thumbnails for ScannerA thumbnails (ID Retrieval Images)

        OOD_Images = []
        for P in OOD_topk_names[idx]:
            OOD_Images.append(PIL.Image.open(os.path.join(args.TMA_ScannerA_Thumb_dir, P[:-11]+".jpg"))) ## Thumbnails for ScannerA thumbnails (OOD Retrieval Images)


    else:
        Q_Img = PIL.Image.open(os.path.join(args.TMA_ScannerB_Thumb_dir, WSI_List[idx][:-11]+".jpg")) ## Thumbnails for ScannerB thumbnails (ID Query Image)
        OOD_Q_Img = PIL.Image.open(os.path.join(args.TMA_ScannerA_Thumb_dir, WSI_List[idx][:-11]+".jpg")) ## Thumbnails for ScannerA thumbnails (OOD Query Image)

        ID_Imgs = []
        for P in topk_names[idx]:
            ID_Imgs.append(PIL.Image.open(os.path.join(args.TMA_ScannerB_Thumb_dir, P[:-11]+".jpg"))) ## Thumbnails for ScannerB thumbnails (ID Retrieval Images)

        OOD_Images = []
        for P in OOD_topk_names[idx]:
            OOD_Images.append(PIL.Image.open(os.path.join(args.TMA_ScannerB_Thumb_dir, P[:-11]+".jpg"))) ## Thumbnails for ScannerB thumbnails (OOD Retrieval Images)



    ID_distances = topk_med_min_distances[idx]
    OOD_distances = OOD_topk_med_min_distances[idx]

    ID_Labels = top5_labels[idx]
    OOD_Labels = OOD_top5_labels[idx]


    # Define border color
    border_colors = {
        'Normal': 'green',
        'Tumour': 'red',
        'NORMAL': 'green',
        'TUMOUR': 'red'
    }

    # Define the color legend patches
    legend_patches = [
        mpatches.Patch(color='green', label='Normal'),
        mpatches.Patch(color='red', label='Tumour')
    ]

    # Create the figure
    fig, axs = plt.subplots(3, 5, figsize=(7, 6))

    # Top row: Query image (centered), leave other slots blank
    for i in range(5):
        axs[0, i].axis('off')  # turn off all
    if Q_Img is not None:
        axs[0, 1].imshow(Q_Img)
        axs[0, 1].set_title('ID Query')
        axs[0, 1].axis('off')
        add_border(axs[0, 1], color=border_colors[WSI_Labels[idx]])

        axs[0, 3].imshow(OOD_Q_Img)
        axs[0, 3].set_title('OOD Query')
        axs[0, 3].axis('off')
        add_border(axs[0, 3], color=border_colors[WSI_Labels[idx]])

    # Second row: ID retrievals
    for i, img in enumerate(ID_Imgs):
        axs[1, i].imshow(img)
        axs[1, i].set_title(f'ID {i+1}, {ID_distances[i]:.2f}')
        axs[1, i].axis('off')
        add_border(axs[1, i], color=border_colors[ID_Labels[i]])

    # Fourth row: OOD retrievals
    for i, img in enumerate(OOD_Images):
        axs[2, i].imshow(img)
        axs[2, i].set_title(f'OOD {i+1}, {OOD_distances[i]:.2f}')
        axs[2, i].axis('off')
        add_border(axs[2, i], color=border_colors[OOD_Labels[i]])


    plt.tight_layout()
    # Add legend here
    plt.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.0, 4.2),
           ncol=1, frameon=False)
    # plt.show()
    plt.savefig(Save_Path, dpi=300)
    plt.close()




# # Main Evaluation Loop
if __name__ == '__main__':
    print("TMA Level Search and Retrieval.\n")
    
    set_seed(args.seed)
    csv_data = pd.read_csv(args.csv_file_dir)

    Network = args.model_name ### For Example, Embeddings from the UNI model is being evaluated

    print(Network)
    ID_Data_Path = args.ID_Data_dir
    OOD_Data_Path = args.OOD_Data_dir

    extensions = ['npy']
    TMA_List = [fn for fn in os.listdir(ID_Data_Path) if any(fn.endswith(ext) for ext in extensions)]

    ID_TMA_List_Embeds = []
    OOD_TMA_List_Embeds = []
    Case_Numbers = []
    Labels = []

    for T in TMA_List:

        String_parts = T.split('-')
        ID_TMA_List_Embeds.append(np.load(os.path.join(ID_Data_Path, T)))
        OOD_TMA_List_Embeds.append(np.load(os.path.join(OOD_Data_Path, T)))
        Labels.append(label_dict[String_parts[2]])
        Case_Numbers.append(get_case_number(String_parts[1]))

    Labels = np.array(Labels)
    TMA_List = np.array(TMA_List)
    #image search using median of minimums
    top1_labels, top3_labels, top5_labels, topk_med_min_distances, topk_names, topk_labels = med_min_search(ID_TMA_List_Embeds, Labels, TMA_List, Case_Numbers)

    #evaluate
    top1_CF = classification_report(y_true=Labels, y_pred=top1_labels)
    top3_CF = classification_report(y_true=Labels, y_pred=top3_labels)
    top5_CF = classification_report(y_true=Labels, y_pred=top5_labels)


    OOD_top1_labels, OOD_top3_labels, OOD_top5_labels, OOD_topk_med_min_distances, OOD_topk_names, OOD_topk_labels = OOD_med_min_search(ID_TMA_List_Embeds, OOD_TMA_List_Embeds, Labels, TMA_List, Case_Numbers)
    #evaluate
    OOD_top1_CF = classification_report(y_true=Labels, y_pred=OOD_top1_labels)
    OOD_top3_CF = classification_report(y_true=Labels, y_pred=OOD_top3_labels)
    OOD_top5_CF = classification_report(y_true=Labels, y_pred=OOD_top5_labels)

    np.save(os.path.join(current_working_directory, ID_Data, Network, 'TMA-Names.npy'), TMA_List)
    np.save(os.path.join(current_working_directory, ID_Data, Network, 'TMA-Top-5_ID-Names.npy'), topk_names)
    np.save(os.path.join(current_working_directory, ID_Data, Network, 'TMA-Top-5_OODD-Names.npy'), OOD_topk_names)

    np.save(os.path.join(current_working_directory, ID_Data, Network, 'ID_TopK-Distances.npy'), topk_med_min_distances)
    np.save(os.path.join(current_working_directory, ID_Data, Network, 'OOD_TopK-Distances.npy'), OOD_topk_med_min_distances)

    draw_top_k_WSIs(TMA_List, topk_names, topk_med_min_distances, topk_labels, OOD_topk_names, OOD_topk_med_min_distances, OOD_topk_labels, Labels, idx=0, Save_Path=os.path.join(current_working_directory, ID_Data, Network, "TMA-Top_5_Retrievals.jpg"))


    # Write the report to a text file
    with open(os.path.join(current_working_directory, ID_Data, Network, 'TMA-Top-1_CR.txt'), 'w') as f:
        f.write("In-Domain\n")
        f.write(top1_CF)
        f.write("\n")
        f.write("Out of Domain\n")
        f.write(OOD_top1_CF)

    with open(os.path.join(current_working_directory, ID_Data, Network, 'TMA-MV-3_CR.txt'), 'w') as f:
        f.write("In-Domain\n")
        f.write(top3_CF)
        f.write("\n")
        f.write("Out of Domain\n")
        f.write(OOD_top3_CF)

    with open(os.path.join(current_working_directory, ID_Data, Network, 'TMA-MV-5_CR.txt'), 'w') as f:
        f.write("In-Domain\n")
        f.write(top5_CF)
        f.write("\n")
        f.write("Out of Domain\n")
        f.write(OOD_top5_CF)
