import os
import numpy as np
from scipy.io import loadmat


def load_results_and_compute_statistics(files_directory, source_name, num_files=5):
    accuracies = []
    classwise_accuracies = []  # This will store class accuracies for each file
    average_accuracies = []  # This will store average accuracies for each file
    kappa_scores = []  # This will store kappa scores for each file

    for i in range(1, num_files + 1):
        file_path = os.path.join(files_directory, 'results_{}times_{}.mat'.format(i, source_name))
        data = loadmat(file_path)
        results = data['results'][0, 0]

        # Extract overall accuracy
        accuracies.append(results['Accuracy'].item())

        # Extract confusion matrix to compute class-wise accuracies
        cm = results['Confusion_matrix']
        class_totals = np.sum(cm, axis=1)  # Total number of instances per class
        class_accuracies = np.diag(cm) / class_totals  # TP / Total per class
        class_accuracies = np.nan_to_num(class_accuracies)  # Convert NaN to zero if any class_totals are zero
        classwise_accuracies.append(class_accuracies)

        # Compute average accuracy for this file
        avg_accuracy = np.nanmean(class_accuracies)
        average_accuracies.append(avg_accuracy)

        # Compute kappa for this file
        total = np.sum(cm)
        pa = np.trace(cm) / float(total)
        pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
        kappa = (pa - pe) / (1 - pe) if (1 - pe) != 0 else 0
        kappa_scores.append(kappa)

    accuracies = np.array(accuracies)
    classwise_accuracies = np.array(classwise_accuracies)
    average_accuracies = np.array(average_accuracies)
    kappa_scores = np.array(kappa_scores)

    # Calculate mean and standard deviation for overall accuracy
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    # Calculate mean and standard deviation for each class accuracy
    mean_classwise_accuracies = np.mean(classwise_accuracies, axis=0)
    std_classwise_accuracies = np.std(classwise_accuracies, axis=0)

    # Calculate mean and standard deviation for average accuracies
    mean_average_accuracy = np.mean(average_accuracies)
    std_average_accuracy = np.std(average_accuracies)

    # Calculate mean and standard deviation for kappa scores
    mean_kappa = np.mean(kappa_scores)
    std_kappa = np.std(kappa_scores)

    print(f"Overall Accuracy (OA): Mean = {mean_accuracy:.3f}%, Std = {std_accuracy:.3f}%")
    print("Classwise Accuracy (AA) per class:")
    for idx, (mean_acc, std_acc) in enumerate(zip(mean_classwise_accuracies, std_classwise_accuracies)):
        print(f"Class {idx}: Mean = {mean_acc:.3f}%, Std = {std_acc:.3f}%")
    print(
        f"Average Accuracy (AA) across all classes: Mean = {mean_average_accuracy:.3f}%, Std = {std_average_accuracy:.3f}%")
    print(f"Overall Kappa: Mean = {mean_kappa:.3f}, Std = {std_kappa:.3f}")

path = r'E:\zts\IEEE_TNNLS_TSTnet\results'
source_name = 'Hangzhou'
load_results_and_compute_statistics(path, source_name)
