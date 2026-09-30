import torch
from torchvision.models import GoogLeNet

import preprocess as pre
import models
import torch.optim as optim
import torch.nn as nn
import os
from sklearn.metrics import accuracy_score, classification_report

def train_model(model, train_loader, task_type, criterion, optimizer, device, epochs=10):
    """Train the model using the given dataset and optimizer."""
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        calculate_accuracy = task_type !="autoencoder"  # Check if accuracy should be calculated

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            if task_type == "binary":
                labels = labels.view(-1, 1).float()
                loss = criterion(outputs, labels)
                predicted = (outputs > 0.5).float()
            elif task_type in ["multi", "comp", "CNN", "VGG", "AlexNet", "ResNet", "GoogleNet", "encoder_resnet"]:
                labels = labels.view(-1).long()  # Ensure correct shape
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
            elif task_type == "autoencoder":
                loss = criterion(outputs, images)
                # Target should be the input images themselves

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Calculate accuracy only for classification tasks
            if calculate_accuracy:
                _, predicted = torch.max(outputs, 1)  # Get predicted labels
                correct += (predicted == labels.squeeze()).sum().item()
                total += labels.size(0)

        if calculate_accuracy:
            accuracy = 100 * correct / total if total > 0 else 0
            print(
                f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")
        else:
            print(
                f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")  # No accuracy for autoencoder


def evaluate_model(model, val_loader, task_type, criterion, device):
    """Evaluate the trained model on validation data."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if task_type == "binary":
                labels = labels.view(-1, 1).float()
                loss = criterion(outputs, labels)
                predicted = (outputs > 0.5).float()
            elif task_type in ["multi", "comp", "CNN", "VGG", "AlexNet","encoder_resnet"]:
                labels = labels.view(-1).long()  # Ensure correct shape
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
            elif task_type == "autoencoder":
                loss = criterion(outputs, images)


            val_loss += loss.item()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    avg_loss = val_loss / len(val_loader)
    print(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return {"loss": avg_loss, "accuracy": accuracy}


# Evaluation function for multiclass classification
def evaluate_random_mini_batches(model, dataloader, num_batches=2):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i >= num_batches:  # Stop after evaluating num_batches
                break
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()  # Use argmax for multiclass predictions
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())  # Ensure labels are numpy arrays

    accuracy = accuracy_score(all_labels, all_preds)  # Calculate accuracy
    print(f"Evaluation on {num_batches} random mini-batches: Accuracy = {accuracy:.2f}%")
    report = classification_report(all_labels, all_preds)  # Generate classification report
    print(report)


def main():
    task_map = {
        "1": "binary",
        "2": "multi",
        "3": "comp",
        "4": "batch",
        "5": "CNN",
        "6": "VGG16",
        "7": "AlexNet",
        "8": "ResNet",
        "9": "GoogLeNet",
        "10":"autoencoder",
        "11": "encoder_resnet"

    }
    accuracies = {
        "binary": [],
        "multi": [],
        "comp": [],
        "batch": [],
        "CNN": [],
        "VGG16": [],
        "AlexNet": [],
        "ResNet": [],
        "GoogLeNet": [],
        "autoencoder": [],
        "encoder_resnet": []
    }

    print("Select Task Type:")
    print("1. Binary Classification")
    print("2. Multiclass Classification")
    print("3. Optimizer Comparison")
    print("4. Random Mini-Batch Evaluation")
    print("5. CNN Model")
    print("6. VGG16 Model")
    print("7. AlexNet Model")
    print("8. ResNet Model")
    print("9. GoogleNet Model")
    print("10. AutoEncoder Model")
    print("11. AutoEncoder with ResNet Classifier")


    task_input = input("Enter choice (1-11): ").strip()
    task_type = task_map.get(task_input)

    if not task_type:
        raise ValueError("Invalid task type. Choose between 1 to 11.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data for all tasks

    path_to_data = "binarydataset" if task_type == "binary" else "Dataset/train"
    # Check if dataset path exists
    if not os.path.exists(path_to_data):
        raise FileNotFoundError(f"Path not found: {path_to_data}")

    train_loader, val_loader, classes = pre.load_data(data_dir=path_to_data, batch_size=32, augment=True)
    print(f"Number of classes: {len(classes)}")
    pre.visualize_data(train_loader, classes, num_samples=5)

    if task_type == "binary":
        model = models.get_binaryclassmodel().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("\nTraining Binary Classification Model...")
        train_model(model, train_loader, task_type, criterion, optimizer, device, epochs=10)
        print("Evaluating Model...")
        evaluate_model(model, val_loader, task_type, criterion, device)

    elif task_type == "multi":
        model = models.get_multiclassmodel(len(classes)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("\nTraining Multiclass Classification Model...")
        train_model(model, train_loader, task_type, criterion, optimizer, device, epochs=10)
        print("Evaluating Model...")
        evaluate_model(model, val_loader, task_type, criterion, device)

    elif task_type == "comp":
        optimizers_list = ["SGD", "Adam", "RMSprop"]
        results = {"optimizers": [], "val_losses": [], "val_accuracies": []}

        for opt_name in optimizers_list:
            print(f"\nTraining with {opt_name} optimizer...")
            model = models.get_multiclassmodel(len(classes)).to(device)
            optimizer = models.get_optimizer(opt_name, model, lr=0.001)
            criterion = nn.CrossEntropyLoss()

            train_model(model, train_loader, task_type, criterion, optimizer, device, epochs=10)
            print(f"Evaluating {opt_name} model...")
            eval_result = evaluate_model(model, val_loader, task_type, criterion, device)

            results["optimizers"].append(opt_name)
            results["val_losses"].append(eval_result["loss"])
            results["val_accuracies"].append(eval_result["accuracy"])

        pre.plot_results(results)

    elif task_type == "batch":
        model = models.get_multiclassmodel(len(classes)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("\nTraining Multiclass Model...")
        train_model(model, train_loader, "multi", criterion, optimizer, device, epochs=10)

        print("Evaluating on Random Mini-Batches...")
        evaluate_random_mini_batches(model, val_loader, num_batches=5)

    elif task_type == "CNN":
        model = models.get_CNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("\nTraining CNN Model...")
        train_model(model, train_loader, "CNN", criterion, optimizer, device, epochs=10)

        print("Evaluating CNN Model...")
        evaluate_model(model, val_loader, "CNN", criterion, device)

    elif task_type == "VGG16":
        model = models.get_VGG16(len(classes)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("\nTraining VGG16 Model...")
        train_model(model, train_loader, "multi", criterion, optimizer, device, epochs=10)

        print("Evaluating VGG16 Model...")
        evaluate_model(model, val_loader, "multi", criterion, device)

    elif task_type == "AlexNet":
        model = models.get_AlexNet(len(classes)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("\nTraining AlexNet Model...")
        train_model(model, train_loader, "multi", criterion, optimizer, device, epochs=10)

        print("Evaluating AlexNet Model...")
        evaluate_model(model, val_loader, "multi", criterion, device)

    elif task_type == "ResNet":
        model = models.get_ResNet(len(classes)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("\nTraining ResNet Model...")
        train_model(model, train_loader, "multi", criterion, optimizer, device, epochs=10)

        print("Evaluating ResNet Model...")
        evaluate_model(model, val_loader, "multi", criterion, device)

    elif task_type == "GoogLeNet":
        model = models.get_GoogLeNet(len(classes)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("\nTraining GoogLeNet Model...")
        train_model(model, train_loader, "multi", criterion, optimizer, device, epochs=10)

        print("Evaluating GoogLeNet Model...")
        evaluate_model(model, val_loader, "multi", criterion, device)

    elif task_type == "autoencoder":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        path_to_data = "dataset"
        train_loader, val_loader, _ = pre.load_data(data_dir=path_to_data, batch_size=32, augment=False)
        model = models.get_autoencodermodel().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("Autoencoder Model Training...")
        train_model(model, train_loader, task_type, criterion, optimizer, device, epochs=10)

        print("Visualizing Autoencoder Results...")
        pre.visualize_autoencoder_output(model, val_loader, device)

    elif task_type == "encoder_resnet":
        model = models.get_encoder_with_resnet(len(classes)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("\nTraining Encoder with ResNet Classifier Model...")
        train_model(model, train_loader, "multi", criterion, optimizer, device, epochs=10)

        print("Evaluating Encoder with ResNet Classifier Model...")
        evaluate_model(model, val_loader, "multi", criterion, device)
        # Plot the model accuracies
        pre.plot_model_accuracies(accuracies)



if __name__ == "__main__":
    main()
