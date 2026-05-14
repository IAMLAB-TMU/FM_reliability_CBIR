import os
import numpy as np
import argparse
import PIL
import random
from scipy.spatial import distance
import bitarray
from bitarray import util as butil

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, balanced_accuracy_score, recall_score

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
import seaborn as sns

parser = argparse.ArgumentParser(description='Patch Level Search and Retrieval')

parser.add_argument('--seed', type=int, default=8)
parser.add_argument('--folds', type=int, default=5)

parser.add_argument('--ID_Query_Embeds_dir', type=str, default="Data Dir for ID Query Embeds")
parser.add_argument('--ID_Query_Labels_dir', type=str, default="Data Dir for ID Query Labels")
parser.add_argument('--ID_Query_Names_dir', type=str, default="Data Dir for ID Query File Names")

parser.add_argument('--OOD_Query_Embeds_dir', type=str, default="Data Dir for OOD Query Embeds")
parser.add_argument('--OOD_Query_Labels_dir', type=str, default="Data Dir for OOD Query Labels")
parser.add_argument('--OOD_Query_Names_dir', type=str, default="Data Dir for OOD Query File Names")

parser.add_argument('--ID_Archive_Embeds_dir', type=str, default="Data Dir for ID Archive Embeds")
parser.add_argument('--ID_Archive_Labels_dir', type=str, default="Data Dir for ID Archive Labels")
parser.add_argument('--ID_Archive_Names_dir', type=str, default="Data Dir for ID Archive File Names")

parser.add_argument('--output_dir', type=str, default=os.getcwd())
parser.add_argument('--model_name', type=str, default='UNI', choices=['UNI', 'Virchow2', 'GigaPath'])

parser.add_argument('--ID_Data', type=str, default="ScannerA", choices=['ScannerA', 'ScannerB'])
parser.add_argument('--OOD_Data', type=str, default="ScannerB", choices=['ScannerA', 'ScannerB'])


args = parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

current_working_dir = args.output_dir

### Define ID and OOD Data
ID_Data = args.ID_Data
OOD_Data = args.OOD_Data

Folds = list(range(1, args.folds + 1)) ### multiple folds dataset

if ID_Data == "ScannerA":
    Results_Folder = "ScannerA_as_ID"
else:
    Results_Folder = "ScannerB_as_ID"


from collections import Counter
from itertools import product

def Majority_Vote(Input_Array):

    # Count occurrences
    counter = Counter(Input_Array)

    # Find the majority string
    majority_Label = counter.most_common(1)[0][0]
    return majority_Label

### code taken from "https://github.com/masih4/MedImageRetrieval/blob/master/metric.py"
def ap_k(actual, predicted, k=5):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    Examples
    --------
    actual, predicted = ['a'], ['a', 'a', 'n', 'a', 'j']
    print(ap_k(actual, predicted, 5))   0.92
     """
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual[:k]:
            # if p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return 0.0 if num_hits == 0.0 else score / num_hits

### code taken from "https://github.com/masih4/MedImageRetrieval/blob/master/metric.py"
def map_k(actual, predicted, k=5):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two list
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    -------
    example :
    actual, predicted = ['p_a', 'p_b'], [
                                        ['p_a', 'p_b', 'p_c', 'p_d', 'p_e', 'p_f'],
                                        ['p_c', 'p_d', 'p_e', 'p_f', 'p_a', 'p_b'],
                                        ['p_d', 'p_a', 'p_c', 'p_b', 'p_e', 'p_f'],
                                        ]
    print(map_k(actual, predicted, 6))
    actual, predicted = ['a'], [['a', 'b', 'a', 'c', 'a', 'c', 'a', 'c', 'c', 'b'],
                                ['c', 'b', 'a', 'v', 'a', 'c', 'a', 'v', 'v', 'v'],
                                ['a', 'x', 'a', 'c', 'a', 'x', 'x', 'c', 'c', 'v'],
                                ['a', 'x', 'a', 'x', 'x', 'c', 'x', 'c', 'c', 'b']]
    print(map_k(actual, predicted, 10))

    print(map_k([1], [[1,1,0,1,0]], k=5)) 0.916
    """
    return np.mean([ap_k(a, p, k) for a, p in product([actual], predicted)])


def normalize_features_numpy(feature_list):
    """
    Normalizes a list of feature vectors using L2 norm.
    Input: List of lists or a 2D NumPy array.
    """
    features = np.array(feature_list)

    # Calculate the L2 norm for each row (axis 1)
    # Use keepdims=True to allow for easy broadcasting (division)
    # Add a tiny epsilon (1e-12) to prevent division by zero for null vectors
    norms = np.linalg.norm(features, ord=2, axis=1, keepdims=True)

    normalized_features = features / (norms + 1e-12)

    return normalized_features


def plot_confusion_matrix_norm(cm, title, output_file, class_names=['Normal', 'Tumour'], Figsize=(6, 5),
                               cellFontSize=18,lineSpace=4.0,fontsizelabel=14,
                               xLabelsRotation=0, tickFontSize=12):

    # Normalize by row sums
    cm_norm_by_row = cm / cm.sum(axis=1, keepdims=True)
    # Convert to percentage
    cm_percentage = np.vectorize(lambda v: f'{v:.1%}')(cm_norm_by_row)
    plt.figure(figsize=Figsize)
    # plt.figure(figsize=(9, 7))
    # Use a larger font size for the tick labels and annotation text

    sns.heatmap(
        cm_norm_by_row,
        annot=cm_percentage,
        fmt='',
        cmap='Blues',  # Use the custom colormap
        annot_kws={'size': cellFontSize, 'fontweight': 'bold'},  # Make the cell font bold
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=lineSpace,  # Add visible separation between cells
        linecolor='white',  # Set the color of the cell separators to white
        vmin=-0.05,  # Set the minimum value for the color scale
        vmax=1.0   # Set the maximum value for the color scale
    )
    plt.title(title, fontsize=fontsizelabel)  # Adjust the font size for the title
    plt.xlabel('Model Diagnosis', fontsize=fontsizelabel)  # Adjust the font size for labels
    plt.ylabel('True Diagnosis', fontsize=fontsizelabel)  # Adjust the font size for labels
    # Increase the font size for tick labels
    tick_label_font = {'fontsize': tickFontSize, 'weight': 'bold'}
    plt.xticks(rotation=xLabelsRotation)
    plt.yticks(rotation=0)
    # Make tick labels bold

    plt.gca().xaxis.set_ticklabels(class_names, fontdict=tick_label_font)
    plt.gca().yaxis.set_ticklabels(class_names, fontdict=tick_label_font)

    # Ensure that the axis labels are fully visible
    plt.tight_layout()
    # plt.show()
    # Save the confusion matrix plot to the output directory
    if title is not None:
        plt.savefig(output_file, dpi=300)
    plt.close()


# Function to draw a colored border around an image
def add_border(ax, color='black', linewidth=6):
    rect = patches.Rectangle((0, 0), 1, 1, linewidth=linewidth, edgecolor=color, facecolor='none', transform=ax.transAxes)
    ax.add_patch(rect)

def Plot_Query_and_Retrieval_Images(query_img, Q_label, OOD_query_img, OOD_Q_label, id_imgs, ID_Labels, ID_distances, ood_imgs, OOD_Labels, OOD_distances, Save_Path):

    # Define border color
    border_colors = {
        'Normal': 'green',
        'Tumour': 'red'
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
    if query_img is not None:
        axs[0, 1].imshow(query_img)
        axs[0, 1].set_title('ID Query')
        axs[0, 1].axis('off')
        add_border(axs[0, 1], color=border_colors[Q_label])

        axs[0, 3].imshow(OOD_query_img)
        axs[0, 3].set_title('OOD Query')
        axs[0, 3].axis('off')
        add_border(axs[0, 3], color=border_colors[OOD_Q_label])

    # Second row: ID retrievals
    for i, img in enumerate(id_imgs):
        axs[1, i].imshow(img)
        axs[1, i].set_title(f'ID {i+1}, {ID_distances[i]:.2f}')
        axs[1, i].axis('off')
        add_border(axs[1, i], color=border_colors[ID_Labels[i]])

    # Fourth row: OOD retrievals
    for i, img in enumerate(ood_imgs):
        axs[2, i].imshow(img)
        axs[2, i].set_title(f'OOD {i+1}, {OOD_distances[i]:.2f}')
        axs[2, i].axis('off')
        add_border(axs[2, i], color=border_colors[OOD_Labels[i]])

    plt.tight_layout()
    # Add legend here
    plt.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 5.0),
           ncol=1, frameon=False)
    # plt.show()
    plt.savefig(Save_Path, dpi=300)
    plt.close()


# # Main Evaluation Loop
if __name__ == '__main__':
    print("Patch Level Search and Retrieval.\n")
    set_seed(args.seed)

    Network = args.model_name ### For Example, Embeddings from the UNI model is being evaluated
    Embed_Norm = False ### if Embeddings are required to be normalized

    for fold in Folds: ## for each fold
        print(f"Fold_{fold}")

        #####Load all the ID data
        ID_Test_Embeds = np.load(os.path.join(args.ID_Query_Embeds_dir, ID_Data, Network, f"Fold_{fold}", "All_Test_Embeds.npy"))
        ID_Test_Labels = np.load(os.path.join(args.ID_Query_Labels_dir, ID_Data, Network, f"Fold_{fold}", "All_Test_Labels.npy"))
        ID_Test_Names = np.load(os.path.join(args.ID_Query_Names_dir, ID_Data, Network, f"Fold_{fold}", "All_Test_Names.npy"))

        # for feat in ID_Test_Embeds: Convert the embeddings into the barcodes
        patch_barcodes = (np.diff(np.array(ID_Test_Embeds), axis=1) < 0)*1
        ID_Test_Barcodes = [bitarray.bitarray(b.tolist()) for b in patch_barcodes]
        if Embed_Norm: ## Normalize Embeds
            ID_Test_Embeds = normalize_features_numpy(ID_Test_Embeds) ## Normalize 

        ID_Train_Embeds = np.load(os.path.join(args.ID_Archive_Embeds_dir, ID_Data, Network, f"Fold_{fold}", "All_Train_Embeds.npy"))
        ID_Train_Labels = np.load(os.path.join(args.ID_Archive_Labels_dir, ID_Data, Network, f"Fold_{fold}", "All_Train_Labels.npy"))
        ID_Train_Names = np.load(os.path.join(args.ID_Archive_Names_dir, ID_Data, Network, f"Fold_{fold}", "All_Train_Names.npy"))

        # for feat in ID_Train_Embeds: Convert the embeddings into the barcodes
        patch_barcodes = (np.diff(np.array(ID_Train_Embeds), axis=1) < 0)*1
        ID_Train_Barcodes = [bitarray.bitarray(b.tolist()) for b in patch_barcodes]
        if Embed_Norm: ## Normalize Embeds 
            ID_Train_Embeds = normalize_features_numpy(ID_Train_Embeds) ## Normalize

        AP_5 = {
            "Euclidean": [],
            "Cosine": [],
            "Hamming": []
        }
        mAP_5 = {
            "Euclidean": [],
            "Cosine": [],
            "Hamming": []
        }

        MV_5 = {
            "Euclidean": [],
            "Cosine": [],
            "Hamming": []
        }

        MV_3 = {
            "Euclidean": [],
            "Cosine": [],
            "Hamming": []
        }

        Top_1 = {
            "Euclidean": [],
            "Cosine": [],
            "Hamming": []
        }

        ID_top_5_Labels = {
            "Euclidean": [],
            "Cosine": [],
            "Hamming": []
        }
        top_5_Names = {
            "Euclidean": [],
            "Cosine": [],
            "Hamming": []
        }
        ID_top_5_Distances = {
            "Euclidean": [],
            "Cosine": [],
            "Hamming": []
        }

        ### MV5, MV3, Top1
        class_metrics = {
            "f1_score-Euclidean": [],
            "f1_score-Cosine": [],
            "f1_score-Hamming": [],
            "recall-Euclidean": [],
            "recall-Cosine": [],
            "recall-Hamming": [],
            "accuracy_score-Euclidean": [],
            "accuracy_score-Cosine": [],
            "accuracy_score-Hamming": [],
            "baccuracy_score-Euclidean": [], ## balanced accuracy
            "baccuracy_score-Cosine": [], ## balanced accuracy
            "baccuracy_score-Hamming": [] ## balanced accuracy
        }


        for iQ, Q in enumerate(ID_Test_Embeds):

            distances = {
            "Euclidean": [],
            "Cosine": [],
            "Hamming": []
            }

            for iA, A in enumerate(ID_Train_Embeds):
                distances["Euclidean"].append(distance.euclidean(Q, A))
                distances["Cosine"].append(distance.cosine(Q, A))
                distances["Hamming"].append(distance.hamming(ID_Test_Barcodes[iQ], ID_Train_Barcodes[iA]))

            sorted_distances_euc = np.argsort(np.array(distances["Euclidean"]))
            sorted_distances_cos = np.argsort(np.array(distances["Cosine"]))
            sorted_distances_ham = np.argsort(np.array(distances["Hamming"]))


            MV_5["Euclidean"].append(Majority_Vote(ID_Train_Labels[sorted_distances_euc[0:5]])) ## Top Five
            MV_5["Cosine"].append(Majority_Vote(ID_Train_Labels[sorted_distances_cos[0:5]])) ## Top Five
            MV_5["Hamming"].append(Majority_Vote(ID_Train_Labels[sorted_distances_ham[0:5]])) ## Top Five

            MV_3["Euclidean"].append(Majority_Vote(ID_Train_Labels[sorted_distances_euc[0:3]])) ## Top Three
            MV_3["Cosine"].append(Majority_Vote(ID_Train_Labels[sorted_distances_cos[0:3]])) ## Top Three
            MV_3["Hamming"].append(Majority_Vote(ID_Train_Labels[sorted_distances_ham[0:3]])) ## Top Three

            Top_1["Euclidean"].append(ID_Train_Labels[sorted_distances_euc[0]]) 
            Top_1["Cosine"].append(ID_Train_Labels[sorted_distances_cos[0]]) 
            Top_1["Hamming"].append(ID_Train_Labels[sorted_distances_ham[0]]) 

            ID_top_5_Labels["Euclidean"].append(ID_Train_Labels[sorted_distances_euc[0:5]])
            ID_top_5_Labels["Cosine"].append(ID_Train_Labels[sorted_distances_cos[0:5]]) 
            ID_top_5_Labels["Hamming"].append(ID_Train_Labels[sorted_distances_ham[0:5]])

            AP_5["Euclidean"].append(ap_k([ID_Test_Labels[iQ]], ID_Train_Labels[sorted_distances_euc[0:5]], k=5))
            mAP_5["Euclidean"].append(map_k(ID_Test_Labels[iQ], ID_Train_Labels[sorted_distances_euc[0:5]], k=5))
            AP_5["Cosine"].append(ap_k([ID_Test_Labels[iQ]], ID_Train_Labels[sorted_distances_cos[0:5]], k=5))
            mAP_5["Cosine"].append(map_k(ID_Test_Labels[iQ], ID_Train_Labels[sorted_distances_cos[0:5]], k=5))
            AP_5["Hamming"].append(ap_k([ID_Test_Labels[iQ]], ID_Train_Labels[sorted_distances_ham[0:5]], k=5))
            mAP_5["Hamming"].append(map_k(ID_Test_Labels[iQ], ID_Train_Labels[sorted_distances_ham[0:5]], k=5))

            top_5_Names["Euclidean"].append(ID_Train_Names[sorted_distances_euc[0:5]])
            top_5_Names["Cosine"].append(ID_Train_Names[sorted_distances_cos[0:5]]) 
            top_5_Names["Hamming"].append(ID_Train_Names[sorted_distances_ham[0:5]])

            ID_top_5_Distances["Euclidean"].append(np.array(distances["Euclidean"])[sorted_distances_euc[0:5]])
            ID_top_5_Distances["Cosine"].append(np.array(distances["Cosine"])[sorted_distances_cos[0:5]]) 
            ID_top_5_Distances["Hamming"].append(np.array(distances["Hamming"])[sorted_distances_ham[0:5]])


        os.makedirs(os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder), exist_ok=True)

        ## Compute ID F1 scores
        class_metrics["f1_score-Euclidean"].append(f1_score(ID_Test_Labels, MV_5["Euclidean"], average='macro'))
        class_metrics["f1_score-Cosine"].append(f1_score(ID_Test_Labels, MV_5["Cosine"], average='macro'))
        class_metrics["f1_score-Hamming"].append(f1_score(ID_Test_Labels, MV_5["Hamming"], average='macro'))

        class_metrics["f1_score-Euclidean"].append(f1_score(ID_Test_Labels, MV_3["Euclidean"], average='macro'))
        class_metrics["f1_score-Cosine"].append(f1_score(ID_Test_Labels, MV_3["Cosine"], average='macro'))
        class_metrics["f1_score-Hamming"].append(f1_score(ID_Test_Labels, MV_3["Hamming"], average='macro'))

        class_metrics["f1_score-Euclidean"].append(f1_score(ID_Test_Labels, Top_1["Euclidean"], average='macro'))
        class_metrics["f1_score-Cosine"].append(f1_score(ID_Test_Labels, Top_1["Cosine"], average='macro'))
        class_metrics["f1_score-Hamming"].append(f1_score(ID_Test_Labels, Top_1["Hamming"], average='macro'))

        ## Compute ID Recall
        class_metrics["recall-Euclidean"].append(recall_score(ID_Test_Labels, MV_5["Euclidean"], average='macro'))
        class_metrics["recall-Cosine"].append(recall_score(ID_Test_Labels, MV_5["Cosine"], average='macro'))
        class_metrics["recall-Hamming"].append(recall_score(ID_Test_Labels, MV_5["Hamming"], average='macro'))

        class_metrics["recall-Euclidean"].append(recall_score(ID_Test_Labels, MV_3["Euclidean"], average='macro'))
        class_metrics["recall-Cosine"].append(recall_score(ID_Test_Labels, MV_3["Cosine"], average='macro'))
        class_metrics["recall-Hamming"].append(recall_score(ID_Test_Labels, MV_3["Hamming"], average='macro'))

        class_metrics["recall-Euclidean"].append(recall_score(ID_Test_Labels, Top_1["Euclidean"], average='macro'))
        class_metrics["recall-Cosine"].append(recall_score(ID_Test_Labels, Top_1["Cosine"], average='macro'))
        class_metrics["recall-Hamming"].append(recall_score(ID_Test_Labels, Top_1["Hamming"], average='macro'))

        ## Compute ID Accuracy Score
        class_metrics["accuracy_score-Euclidean"].append(accuracy_score(ID_Test_Labels, MV_5["Euclidean"]))
        class_metrics["accuracy_score-Cosine"].append(accuracy_score(ID_Test_Labels, MV_5["Cosine"]))
        class_metrics["accuracy_score-Hamming"].append(accuracy_score(ID_Test_Labels, MV_5["Hamming"]))

        class_metrics["accuracy_score-Euclidean"].append(accuracy_score(ID_Test_Labels, MV_3["Euclidean"]))
        class_metrics["accuracy_score-Cosine"].append(accuracy_score(ID_Test_Labels, MV_3["Cosine"]))
        class_metrics["accuracy_score-Hamming"].append(accuracy_score(ID_Test_Labels, MV_3["Hamming"]))

        class_metrics["accuracy_score-Euclidean"].append(accuracy_score(ID_Test_Labels, Top_1["Euclidean"]))
        class_metrics["accuracy_score-Cosine"].append(accuracy_score(ID_Test_Labels, Top_1["Cosine"]))
        class_metrics["accuracy_score-Hamming"].append(accuracy_score(ID_Test_Labels, Top_1["Hamming"]))

        ## Compute ID Balanced Accuracy Score
        class_metrics["baccuracy_score-Euclidean"].append(balanced_accuracy_score(ID_Test_Labels, MV_5["Euclidean"]))
        class_metrics["baccuracy_score-Cosine"].append(balanced_accuracy_score(ID_Test_Labels, MV_5["Cosine"]))
        class_metrics["baccuracy_score-Hamming"].append(balanced_accuracy_score(ID_Test_Labels, MV_5["Hamming"]))

        class_metrics["baccuracy_score-Euclidean"].append(balanced_accuracy_score(ID_Test_Labels, MV_3["Euclidean"]))
        class_metrics["baccuracy_score-Cosine"].append(balanced_accuracy_score(ID_Test_Labels, MV_3["Cosine"]))
        class_metrics["baccuracy_score-Hamming"].append(balanced_accuracy_score(ID_Test_Labels, MV_3["Hamming"]))

        class_metrics["baccuracy_score-Euclidean"].append(balanced_accuracy_score(ID_Test_Labels, Top_1["Euclidean"]))
        class_metrics["baccuracy_score-Cosine"].append(balanced_accuracy_score(ID_Test_Labels, Top_1["Cosine"]))
        class_metrics["baccuracy_score-Hamming"].append(balanced_accuracy_score(ID_Test_Labels, Top_1["Hamming"]))

        np.save(os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "ID_class_metrics_MV5-MV3-Top1.npy"), class_metrics)

        ## Compute Classification Reports
        top_1_report_euc = classification_report(ID_Test_Labels, Top_1["Euclidean"])
        top_1_cm_euc = confusion_matrix(ID_Test_Labels, Top_1["Euclidean"])
        plot_confusion_matrix_norm(top_1_cm_euc, title="ID Top 1 (Euclidean Distance)", output_file=os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "ID_Top_1_CM_Euclidean.jpg"))
        top_1_report_cos = classification_report(ID_Test_Labels, Top_1["Cosine"])
        top_1_cm_cos = confusion_matrix(ID_Test_Labels, Top_1["Cosine"])
        plot_confusion_matrix_norm(top_1_cm_cos, title="ID Top 1 (Cosine Distance)", output_file=os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "ID_Top_1_CM_Cosine.jpg"))
        top_1_report_ham = classification_report(ID_Test_Labels, Top_1["Hamming"])
        top_1_cm_ham = confusion_matrix(ID_Test_Labels, Top_1["Hamming"])
        plot_confusion_matrix_norm(top_1_cm_ham, title="ID Top 1 (Hamming Distance)", output_file=os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "ID_Top_1_CM_Hamming.jpg"))

        MV_3_report_euc = classification_report(ID_Test_Labels, MV_3["Euclidean"])
        MV_3_cm_euc = confusion_matrix(ID_Test_Labels, MV_3["Euclidean"])
        plot_confusion_matrix_norm(MV_3_cm_euc, title="ID MV @ Top 3 (Euclidean Distance)", output_file=os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "ID_MV_3_CM_Euclidean.jpg"))
        MV_3_report_cos = classification_report(ID_Test_Labels, MV_3["Cosine"])
        MV_3_cm_cos = confusion_matrix(ID_Test_Labels, MV_3["Cosine"])
        plot_confusion_matrix_norm(MV_3_cm_cos, title="ID MV @ Top 3 (Cosine Distance)", output_file=os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "ID_MV_3_CM_Cosine.jpg"))
        MV_3_report_ham = classification_report(ID_Test_Labels, MV_3["Hamming"])
        MV_3_cm_ham = confusion_matrix(ID_Test_Labels, MV_3["Hamming"])
        plot_confusion_matrix_norm(MV_3_cm_ham, title="ID MV @ Top 3 (Hamming Distance)", output_file=os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "ID_MV_3_CM_Hamming.jpg"))

        MV_5_report_euc = classification_report(ID_Test_Labels, MV_5["Euclidean"])
        MV_5_cm_euc = confusion_matrix(ID_Test_Labels, MV_5["Euclidean"])
        plot_confusion_matrix_norm(MV_5_cm_euc, title="ID MV @ Top 5 (Euclidean Distance)", output_file=os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "ID_MV_5_CM_Euclidean.jpg"))
        MV_5_report_cos = classification_report(ID_Test_Labels, MV_5["Cosine"])
        MV_5_cm_cos = confusion_matrix(ID_Test_Labels, MV_5["Cosine"])
        plot_confusion_matrix_norm(MV_5_cm_cos, title="ID MV @ Top 5 (Cosine Distance)", output_file=os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "ID_MV_5_CM_Cosine.jpg"))
        MV_5_report_ham = classification_report(ID_Test_Labels, MV_5["Hamming"])
        MV_5_cm_ham = confusion_matrix(ID_Test_Labels, MV_5["Hamming"])
        plot_confusion_matrix_norm(MV_5_cm_ham, title="ID MV @ Top 5 (Hamming Distance)", output_file=os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "ID_MV_5_CM_Hamming.jpg"))


        np.save(os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "ID_Top_5_Names.npy"), top_5_Names)
        np.save(os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "ID_Top_5_Distances.npy"), ID_top_5_Distances)

        np.save(os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "ID_AP_5.npy"), AP_5)
        np.save(os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "ID_mAP_5.npy"), mAP_5)

        # Write the report to a text file
        with open(os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "ID_Top_1.txt"), 'w') as R_writer:
            R_writer.write("Euclidean Distance\n")
            R_writer.write(top_1_report_euc)
            R_writer.write("\n")
            R_writer.write("Cosine Distance\n")
            R_writer.write(top_1_report_cos)
            R_writer.write("\n")
            R_writer.write("Hamming Distance\n")
            R_writer.write(top_1_report_ham)
            R_writer.write("\n\n\n")
            R_writer.write('Confusion Matrix:\n')
            R_writer.write('Labels: Normal, Tumour\n')
            R_writer.write("Euclidean Distance CM\n")
            np.savetxt(R_writer, top_1_cm_euc, fmt='%d')
            R_writer.write("Cosine Distance CM\n")
            np.savetxt(R_writer, top_1_cm_cos, fmt='%d')
            R_writer.write("Hamming Distance CM\n")
            np.savetxt(R_writer, top_1_cm_ham, fmt='%d')

        with open(os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "ID_MV_3.txt"), 'w') as R_writer:
            R_writer.write("Euclidean Distance\n")
            R_writer.write(MV_3_report_euc)
            R_writer.write("\n")
            R_writer.write("Cosine Distance\n")
            R_writer.write(MV_3_report_cos)
            R_writer.write("\n")
            R_writer.write("Hamming Distance\n")
            R_writer.write(MV_3_report_ham)
            R_writer.write("\n\n\n")
            R_writer.write('Confusion Matrix:\n')
            R_writer.write('Labels: Normal, Tumour\n')
            R_writer.write("Euclidean Distance CM\n")
            np.savetxt(R_writer, MV_3_cm_euc, fmt='%d')
            R_writer.write("Cosine Distance CM\n")
            np.savetxt(R_writer, MV_3_cm_cos, fmt='%d')
            R_writer.write("Hamming Distance CM\n")
            np.savetxt(R_writer, MV_3_cm_ham, fmt='%d')

        with open(os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "ID_MV_5.txt"), 'w') as R_writer:
            R_writer.write("Euclidean Distance\n")
            R_writer.write(MV_5_report_euc)
            R_writer.write("\n")
            R_writer.write("Cosine Distance\n")
            R_writer.write(MV_5_report_cos)
            R_writer.write("\n")
            R_writer.write("Hamming Distance\n")
            R_writer.write(MV_5_report_ham)
            R_writer.write("\n\n\n")
            R_writer.write('Confusion Matrix:\n')
            R_writer.write('Labels: Normal, Tumour\n')
            R_writer.write("Euclidean Distance CM\n")
            np.savetxt(R_writer, MV_5_cm_euc, fmt='%d')
            R_writer.write("Cosine Distance CM\n")
            np.savetxt(R_writer, MV_5_cm_cos, fmt='%d')
            R_writer.write("Hamming Distance CM\n")
            np.savetxt(R_writer, MV_5_cm_ham, fmt='%d')


        ID_top_5_Names = top_5_Names

        ID_Images_euc = []
        for P in top_5_Names["Euclidean"][0]:
            ID_Images_euc.append(PIL.Image.open(P))

        ID_Images_cos = []
        for P in top_5_Names["Cosine"][0]:
            ID_Images_cos.append(PIL.Image.open(P))

        ID_Images_ham = []
        for P in top_5_Names["Hamming"][0]:
            ID_Images_ham.append(PIL.Image.open(P))

        ## Clear Memory
        del top_1_report_euc, top_1_report_cos, top_1_report_ham, MV_3_report_euc, MV_3_report_cos, MV_3_report_ham, MV_5_report_euc, MV_5_report_cos, MV_5_report_ham, top_5_Names, Top_1, MV_3, MV_5, distances, top_1_cm_euc, top_1_cm_cos, top_1_cm_ham, MV_3_cm_euc, MV_3_cm_cos, MV_3_cm_ham, MV_5_cm_euc, MV_5_cm_cos, MV_5_cm_ham
        del mAP_5, class_metrics, AP_5

        ### OOD
        #### Load All OOD Data
        OOD_Test_Embeds = np.load(os.path.join(args.OOD_Query_Embeds_dir, OOD_Data, Network, f"Fold_{fold}", "All_Test_Embeds.npy"))
        OOD_Test_Labels = np.load(os.path.join(args.OOD_Query_Labels_dir, OOD_Data, Network, f"Fold_{fold}", "All_Test_Labels.npy"))
        OOD_Test_Names = np.load(os.path.join(args.OOD_Query_Names_dir, OOD_Data, Network, f"Fold_{fold}", "All_Test_Names.npy"))

        # for feat in OOD_Test_Embeds: Convert the embeddings into the barcodes
        patch_barcodes = (np.diff(np.array(OOD_Test_Embeds), axis=1) < 0)*1
        OOD_Test_Barcodes = [bitarray.bitarray(b.tolist()) for b in patch_barcodes]
        if Embed_Norm: ## Normalize Embeds
            OOD_Test_Embeds = normalize_features_numpy(OOD_Test_Embeds) #Normalize

        AP_5 = {
            "Euclidean": [],
            "Cosine": [],
            "Hamming": []
        }
        mAP_5 = {
            "Euclidean": [],
            "Cosine": [],
            "Hamming": []
        }

        MV_5 = {
            "Euclidean": [],
            "Cosine": [],
            "Hamming": []
        }

        MV_3 = {
            "Euclidean": [],
            "Cosine": [],
            "Hamming": []
        }

        Top_1 = {
            "Euclidean": [],
            "Cosine": [],
            "Hamming": []
        }

        OOD_top_5_Labels = {
            "Euclidean": [],
            "Cosine": [],
            "Hamming": []
        }
        top_5_Names = {
            "Euclidean": [],
            "Cosine": [],
            "Hamming": []
        }
        OOD_top_5_Distances = {
            "Euclidean": [],
            "Cosine": [],
            "Hamming": []
        }

        ### MV5, MV3, Top1
        class_metrics = {
            "f1_score-Euclidean": [],
            "f1_score-Cosine": [],
            "f1_score-Hamming": [],
            "recall-Euclidean": [],
            "recall-Cosine": [],
            "recall-Hamming": [],
            "accuracy_score-Euclidean": [],
            "accuracy_score-Cosine": [],
            "accuracy_score-Hamming": [],
            "baccuracy_score-Euclidean": [], ## balanced accuracy
            "baccuracy_score-Cosine": [], ## balanced accuracy
            "baccuracy_score-Hamming": [] ## balanced accuracy
        }

        for iQ, Q in enumerate(OOD_Test_Embeds):

            distances = {
            "Euclidean": [],
            "Cosine": [],
            "Hamming": []
            }
            for iA, A in enumerate(ID_Train_Embeds):
                distances["Euclidean"].append(distance.euclidean(Q, A))
                distances["Cosine"].append(distance.cosine(Q, A))
                distances["Hamming"].append(distance.hamming(OOD_Test_Barcodes[iQ], ID_Train_Barcodes[iA]))

            sorted_distances_euc = np.argsort(np.array(distances["Euclidean"]))
            sorted_distances_cos = np.argsort(np.array(distances["Cosine"]))
            sorted_distances_ham = np.argsort(np.array(distances["Hamming"]))


            MV_5["Euclidean"].append(Majority_Vote(ID_Train_Labels[sorted_distances_euc[0:5]])) ## First Five
            MV_5["Cosine"].append(Majority_Vote(ID_Train_Labels[sorted_distances_cos[0:5]])) ## Last Five
            MV_5["Hamming"].append(Majority_Vote(ID_Train_Labels[sorted_distances_ham[0:5]])) ## First Five

            MV_3["Euclidean"].append(Majority_Vote(ID_Train_Labels[sorted_distances_euc[0:3]])) ## First Three
            MV_3["Cosine"].append(Majority_Vote(ID_Train_Labels[sorted_distances_cos[0:3]])) ## Last Three
            MV_3["Hamming"].append(Majority_Vote(ID_Train_Labels[sorted_distances_ham[0:3]])) ## First Three

            Top_1["Euclidean"].append(ID_Train_Labels[sorted_distances_euc[0]]) ## First
            Top_1["Cosine"].append(ID_Train_Labels[sorted_distances_cos[0]]) ## Last
            Top_1["Hamming"].append(ID_Train_Labels[sorted_distances_ham[0]]) ## First

            OOD_top_5_Labels["Euclidean"].append(ID_Train_Labels[sorted_distances_euc[0:5]])
            OOD_top_5_Labels["Cosine"].append(ID_Train_Labels[sorted_distances_cos[0:5]]) ## Last Five
            OOD_top_5_Labels["Hamming"].append(ID_Train_Labels[sorted_distances_ham[0:5]])

            AP_5["Euclidean"].append(ap_k([ID_Test_Labels[iQ]], ID_Train_Labels[sorted_distances_euc[0:5]], k=5))
            mAP_5["Euclidean"].append(map_k(ID_Test_Labels[iQ], ID_Train_Labels[sorted_distances_euc[0:5]], k=5))
            AP_5["Cosine"].append(ap_k([ID_Test_Labels[iQ]], ID_Train_Labels[sorted_distances_cos[0:5]], k=5))
            mAP_5["Cosine"].append(map_k(ID_Test_Labels[iQ], ID_Train_Labels[sorted_distances_cos[0:5]], k=5))
            AP_5["Hamming"].append(ap_k([ID_Test_Labels[iQ]], ID_Train_Labels[sorted_distances_ham[0:5]], k=5))
            mAP_5["Hamming"].append(map_k(ID_Test_Labels[iQ], ID_Train_Labels[sorted_distances_ham[0:5]], k=5))

            top_5_Names["Euclidean"].append(ID_Train_Names[sorted_distances_euc[0:5]])
            top_5_Names["Cosine"].append(ID_Train_Names[sorted_distances_cos[0:5]]) ## Last Five
            top_5_Names["Hamming"].append(ID_Train_Names[sorted_distances_ham[0:5]])

            OOD_top_5_Distances["Euclidean"].append(np.array(distances["Euclidean"])[sorted_distances_euc[0:5]])
            OOD_top_5_Distances["Cosine"].append(np.array(distances["Cosine"])[sorted_distances_cos[0:5]]) ## Last Five
            OOD_top_5_Distances["Hamming"].append(np.array(distances["Hamming"])[sorted_distances_ham[0:5]])


        ## Compute ID F1 scores
        class_metrics["f1_score-Euclidean"].append(f1_score(ID_Test_Labels, MV_5["Euclidean"], average='macro'))
        class_metrics["f1_score-Cosine"].append(f1_score(ID_Test_Labels, MV_5["Cosine"], average='macro'))
        class_metrics["f1_score-Hamming"].append(f1_score(ID_Test_Labels, MV_5["Hamming"], average='macro'))

        class_metrics["f1_score-Euclidean"].append(f1_score(ID_Test_Labels, MV_3["Euclidean"], average='macro'))
        class_metrics["f1_score-Cosine"].append(f1_score(ID_Test_Labels, MV_3["Cosine"], average='macro'))
        class_metrics["f1_score-Hamming"].append(f1_score(ID_Test_Labels, MV_3["Hamming"], average='macro'))

        class_metrics["f1_score-Euclidean"].append(f1_score(ID_Test_Labels, Top_1["Euclidean"], average='macro'))
        class_metrics["f1_score-Cosine"].append(f1_score(ID_Test_Labels, Top_1["Cosine"], average='macro'))
        class_metrics["f1_score-Hamming"].append(f1_score(ID_Test_Labels, Top_1["Hamming"], average='macro'))

        ## Compute ID Recall
        class_metrics["recall-Euclidean"].append(recall_score(ID_Test_Labels, MV_5["Euclidean"], average='macro'))
        class_metrics["recall-Cosine"].append(recall_score(ID_Test_Labels, MV_5["Cosine"], average='macro'))
        class_metrics["recall-Hamming"].append(recall_score(ID_Test_Labels, MV_5["Hamming"], average='macro'))

        class_metrics["recall-Euclidean"].append(recall_score(ID_Test_Labels, MV_3["Euclidean"], average='macro'))
        class_metrics["recall-Cosine"].append(recall_score(ID_Test_Labels, MV_3["Cosine"], average='macro'))
        class_metrics["recall-Hamming"].append(recall_score(ID_Test_Labels, MV_3["Hamming"], average='macro'))

        class_metrics["recall-Euclidean"].append(recall_score(ID_Test_Labels, Top_1["Euclidean"], average='macro'))
        class_metrics["recall-Cosine"].append(recall_score(ID_Test_Labels, Top_1["Cosine"], average='macro'))
        class_metrics["recall-Hamming"].append(recall_score(ID_Test_Labels, Top_1["Hamming"], average='macro'))

        ## Compute ID Accuracy Score
        class_metrics["accuracy_score-Euclidean"].append(accuracy_score(ID_Test_Labels, MV_5["Euclidean"]))
        class_metrics["accuracy_score-Cosine"].append(accuracy_score(ID_Test_Labels, MV_5["Cosine"]))
        class_metrics["accuracy_score-Hamming"].append(accuracy_score(ID_Test_Labels, MV_5["Hamming"]))

        class_metrics["accuracy_score-Euclidean"].append(accuracy_score(ID_Test_Labels, MV_3["Euclidean"]))
        class_metrics["accuracy_score-Cosine"].append(accuracy_score(ID_Test_Labels, MV_3["Cosine"]))
        class_metrics["accuracy_score-Hamming"].append(accuracy_score(ID_Test_Labels, MV_3["Hamming"]))

        class_metrics["accuracy_score-Euclidean"].append(accuracy_score(ID_Test_Labels, Top_1["Euclidean"]))
        class_metrics["accuracy_score-Cosine"].append(accuracy_score(ID_Test_Labels, Top_1["Cosine"]))
        class_metrics["accuracy_score-Hamming"].append(accuracy_score(ID_Test_Labels, Top_1["Hamming"]))

        ## Compute ID Balanced Accuracy Score
        class_metrics["baccuracy_score-Euclidean"].append(balanced_accuracy_score(ID_Test_Labels, MV_5["Euclidean"]))
        class_metrics["baccuracy_score-Cosine"].append(balanced_accuracy_score(ID_Test_Labels, MV_5["Cosine"]))
        class_metrics["baccuracy_score-Hamming"].append(balanced_accuracy_score(ID_Test_Labels, MV_5["Hamming"]))

        class_metrics["baccuracy_score-Euclidean"].append(balanced_accuracy_score(ID_Test_Labels, MV_3["Euclidean"]))
        class_metrics["baccuracy_score-Cosine"].append(balanced_accuracy_score(ID_Test_Labels, MV_3["Cosine"]))
        class_metrics["baccuracy_score-Hamming"].append(balanced_accuracy_score(ID_Test_Labels, MV_3["Hamming"]))

        class_metrics["baccuracy_score-Euclidean"].append(balanced_accuracy_score(ID_Test_Labels, Top_1["Euclidean"]))
        class_metrics["baccuracy_score-Cosine"].append(balanced_accuracy_score(ID_Test_Labels, Top_1["Cosine"]))
        class_metrics["baccuracy_score-Hamming"].append(balanced_accuracy_score(ID_Test_Labels, Top_1["Hamming"]))

        np.save(os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "OOD_class_metrics_MV5-MV3-Top1.npy"), class_metrics)

        top_1_report_euc = classification_report(ID_Test_Labels, Top_1["Euclidean"])
        top_1_cm_euc = confusion_matrix(ID_Test_Labels, Top_1["Euclidean"])
        plot_confusion_matrix_norm(top_1_cm_euc, title="OOD Top 1 (Euclidean Distance)", output_file=os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "OOD_Top_1_CM_Euclidean.jpg"))
        top_1_report_cos = classification_report(ID_Test_Labels, Top_1["Cosine"])
        top_1_cm_cos = confusion_matrix(ID_Test_Labels, Top_1["Cosine"])
        plot_confusion_matrix_norm(top_1_cm_cos, title="OOD Top 1 (Cosine Distance)", output_file=os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "OOD_Top_1_CM_Cosine.jpg"))
        top_1_report_ham = classification_report(ID_Test_Labels, Top_1["Hamming"])
        top_1_cm_ham = confusion_matrix(ID_Test_Labels, Top_1["Hamming"])
        plot_confusion_matrix_norm(top_1_cm_ham, title="OOD Top 1 (Hamming Distance)", output_file=os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "OOD_Top_1_CM_Hamming.jpg"))

        MV_3_report_euc = classification_report(ID_Test_Labels, MV_3["Euclidean"])
        MV_3_cm_euc = confusion_matrix(ID_Test_Labels, MV_3["Euclidean"])
        plot_confusion_matrix_norm(MV_3_cm_euc, title="OOD MV @ Top 3 (Euclidean Distance)", output_file=os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "OOD_MV_3_CM_Euclidean.jpg"))
        MV_3_report_cos = classification_report(ID_Test_Labels, MV_3["Cosine"])
        MV_3_cm_cos = confusion_matrix(ID_Test_Labels, MV_3["Cosine"])
        plot_confusion_matrix_norm(MV_3_cm_cos, title="OOD MV @ Top 3 (Cosine Distance)", output_file=os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "OOD_MV_3_CM_Cosine.jpg"))
        MV_3_report_ham = classification_report(ID_Test_Labels, MV_3["Hamming"])
        MV_3_cm_ham = confusion_matrix(ID_Test_Labels, MV_3["Hamming"])
        plot_confusion_matrix_norm(MV_3_cm_ham, title="OOD MV @ Top 3 (Hamming Distance)", output_file=os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "OOD_MV_3_CM_Hamming.jpg"))

        MV_5_report_euc = classification_report(ID_Test_Labels, MV_5["Euclidean"])
        MV_5_cm_euc = confusion_matrix(ID_Test_Labels, MV_5["Euclidean"])
        plot_confusion_matrix_norm(MV_5_cm_euc, title="OOD MV @ Top 5 (Euclidean Distance)", output_file=os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "OOD_MV_5_CM_Euclidean.jpg"))
        MV_5_report_cos = classification_report(ID_Test_Labels, MV_5["Cosine"])
        MV_5_cm_cos = confusion_matrix(ID_Test_Labels, MV_5["Cosine"])
        plot_confusion_matrix_norm(MV_5_cm_cos, title="OOD MV @ Top 5 (Cosine Distance)", output_file=os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "OOD_MV_5_CM_Cosine.jpg"))
        MV_5_report_ham = classification_report(ID_Test_Labels, MV_5["Hamming"])
        MV_5_cm_ham = confusion_matrix(ID_Test_Labels, MV_5["Hamming"])
        plot_confusion_matrix_norm(MV_5_cm_ham, title="OOD MV @ Top 5 (Hamming Distance)", output_file=os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "OOD_MV_5_CM_Hamming.jpg"))


        np.save(os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "OOD_Top_5_Names.npy"), top_5_Names)
        np.save(os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "OOD_Top_5_Distances.npy"), OOD_top_5_Distances)

        np.save(os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "OOD_AP_5.npy"), AP_5)
        np.save(os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "OOD_mAP_5.npy"), mAP_5)

        # Write the report to a text file
        with open(os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "OOD_Top_1.txt"), 'w') as R_writer:
            R_writer.write("Euclidean Distance\n")
            R_writer.write(top_1_report_euc)
            R_writer.write("\n")
            R_writer.write("Cosine Distance\n")
            R_writer.write(top_1_report_cos)
            R_writer.write("\n")
            R_writer.write("Hamming Distance\n")
            R_writer.write(top_1_report_ham)
            R_writer.write("\n")
            R_writer.write("\n\n\n")
            R_writer.write('Confusion Matrix:\n')
            R_writer.write('Labels: Normal, Tumour\n')
            R_writer.write("Euclidean Distance CM\n")
            np.savetxt(R_writer, top_1_cm_euc, fmt='%d')
            R_writer.write("Cosine Distance CM\n")
            np.savetxt(R_writer, top_1_cm_cos, fmt='%d')
            R_writer.write("Hamming Distance CM\n")
            np.savetxt(R_writer, top_1_cm_ham, fmt='%d')

        with open(os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "OOD_MV_3.txt"), 'w') as R_writer:
            R_writer.write("Euclidean Distance\n")
            R_writer.write(MV_3_report_euc)
            R_writer.write("Cosine Distance\n")
            R_writer.write(MV_3_report_cos)
            R_writer.write("Hamming Distance\n")
            R_writer.write(MV_3_report_ham)
            R_writer.write("\n\n\n")
            R_writer.write('Confusion Matrix:\n')
            R_writer.write('Labels: Normal, Tumour\n')
            R_writer.write("Euclidean Distance CM\n")
            np.savetxt(R_writer, MV_3_cm_euc, fmt='%d')
            R_writer.write("Cosine Distance CM\n")
            np.savetxt(R_writer, MV_3_cm_cos, fmt='%d')
            R_writer.write("Hamming Distance CM\n")
            np.savetxt(R_writer, MV_3_cm_ham, fmt='%d')

        with open(os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "OOD_MV_5.txt"), 'w') as R_writer:
            R_writer.write("Euclidean Distance\n")
            R_writer.write(MV_5_report_euc)
            R_writer.write("Cosine Distance\n")
            R_writer.write(MV_5_report_cos)
            R_writer.write("Hamming Distance\n")
            R_writer.write(MV_5_report_ham)
            R_writer.write("\n\n\n")
            R_writer.write('Confusion Matrix:\n')
            R_writer.write('Labels: Normal, Tumour\n')
            R_writer.write("Euclidean Distance CM\n")
            np.savetxt(R_writer, MV_5_cm_euc, fmt='%d')
            R_writer.write("Cosine Distance CM\n")
            np.savetxt(R_writer, MV_5_cm_cos, fmt='%d')
            R_writer.write("Hamming Distance CM\n")
            np.savetxt(R_writer, MV_5_cm_ham, fmt='%d')

        ## Plot Sample Image
        Q_Img = PIL.Image.open(ID_Test_Names[0])
        OOD_Q_Img = PIL.Image.open(OOD_Test_Names[0])

        ### Set the Path for OOD Corresponding Patches given that the corresponding patches are stored in the path that differes in the scanner names only
        if ID_Data == "ScannerA":
            ID_OOD_top_5_Names_euc = [path.replace("ScannerA", "ScannerB") for path in ID_top_5_Names["Euclidean"][0]]
            ID_OOD_top_5_Names_cos = [path.replace("ScannerA", "ScannerB") for path in ID_top_5_Names["Cosine"][0]]
            ID_OOD_top_5_Names_ham = [path.replace("ScannerA", "ScannerB") for path in ID_top_5_Names["Hamming"][0]]
        else:
            ID_OOD_top_5_Names_euc = [path.replace("ScannerB", "ScannerA") for path in ID_top_5_Names["Euclidean"][0]]
            ID_OOD_top_5_Names_cos = [path.replace("ScannerB", "ScannerA") for path in ID_top_5_Names["Cosine"][0]]
            ID_OOD_top_5_Names_ham = [path.replace("ScannerB", "ScannerA") for path in ID_top_5_Names["Hamming"][0]]

        ID_OOD_Imgs_euc = []
        ID_OOD_Imgs_cos = []
        ID_OOD_Imgs_ham = []
        for i, P in enumerate(ID_OOD_top_5_Names_euc):
            ID_OOD_Imgs_euc.append(PIL.Image.open(P))
            ID_OOD_Imgs_cos.append(PIL.Image.open(ID_OOD_top_5_Names_cos[i]))
            ID_OOD_Imgs_ham.append(PIL.Image.open(ID_OOD_top_5_Names_cos[i]))

        OOD_Images_euc = []
        for P in top_5_Names["Euclidean"][0]:
            OOD_Images_euc.append(PIL.Image.open(P))

        OOD_Images_cos = []
        for P in top_5_Names["Cosine"][0]:
            OOD_Images_cos.append(PIL.Image.open(P))

        OOD_Images_ham = []
        for P in top_5_Names["Hamming"][0]:
            OOD_Images_ham.append(PIL.Image.open(P))

        Plot_Query_and_Retrieval_Images(Q_Img, ID_Test_Labels[0], OOD_Q_Img, OOD_Test_Labels[0], ID_OOD_Imgs_euc, ID_top_5_Labels["Euclidean"][0], ID_top_5_Distances["Euclidean"][0], OOD_Images_euc, OOD_top_5_Labels["Euclidean"][0], OOD_top_5_Distances["Euclidean"][0], Save_Path=os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "Top_5_Retrievals_Euclidean.jpg"))
        Plot_Query_and_Retrieval_Images(Q_Img, ID_Test_Labels[0], OOD_Q_Img, OOD_Test_Labels[0], ID_OOD_Imgs_cos, ID_top_5_Labels["Cosine"][0], ID_top_5_Distances["Cosine"][0], OOD_Images_cos, OOD_top_5_Labels["Cosine"][0], OOD_top_5_Distances["Cosine"][0], Save_Path=os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "Top_5_Retrievals_Cosine.jpg"))
        Plot_Query_and_Retrieval_Images(Q_Img, ID_Test_Labels[0], OOD_Q_Img, OOD_Test_Labels[0], ID_OOD_Imgs_ham, ID_top_5_Labels["Hamming"][0], ID_top_5_Distances["Hamming"][0], OOD_Images_ham, OOD_top_5_Labels["Hamming"][0], OOD_top_5_Distances["Hamming"][0], Save_Path=os.path.join(current_working_dir, ID_Data, Network, f"Fold_{fold}", Results_Folder, "Top_5_Retrievals_Hamming.jpg"))


