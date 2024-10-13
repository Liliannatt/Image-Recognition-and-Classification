
from load_data import load_data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from net import Net, Net_A1, Net_A2, Net_B, Net_C, Net_D, Net_E, Net_5A
import copy
import time

# ok
def train_net(net, trainloader, valloader, optimizer, criterion, epochs=300, patience=20):

    net.train()
    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(epochs):
        running_loss = 0.0
        net.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_acc = evaluate_net(net, valloader)
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.4f}, Val Acc: {val_acc * 100:.2f}%')

        # 早停机制
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(net.state_dict())
        #     epochs_no_improve = 0
        # else:
        #     epochs_no_improve += 1
        #     if epochs_no_improve >= patience:
        #         print(f'Early stopping at epoch {epoch + 1}')
        #         break

    net.load_state_dict(best_model_wts)
    print('Finished Training')


def evaluate_net(net, dataloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


def test_net(net, testloader):
    net.eval()
    num_classes = 10
    true_positives = [0] * num_classes
    false_negatives = [0] * num_classes

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            for cls in range(num_classes):
                true_positives[cls] += ((predicted == cls) & (labels == cls)).sum().item()
                false_negatives[cls] += ((predicted != cls) & (labels == cls)).sum().item()

    recall_per_class = []
    for cls in range(num_classes):
        tp = true_positives[cls]
        fn = false_negatives[cls]
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0
        recall_per_class.append(recall)
        print(f'Recall of class {cls}: {recall * 100:.2f}%')

    average_recall = sum(recall_per_class) / num_classes
    print(f'Average Top-1 Recall Rate: {average_recall * 100:.2f}%')
    return average_recall

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    criterion = nn.CrossEntropyLoss()

    print("******************************************")
    print("********  normal (batch size = 64, learning rate = 0.001, No Shuffle)  ********")
    print("******************************************")
    net = Net_5A()
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=1e-3)
    trainloader, testloader, valloader = load_data(batch_size=64, num_workers=4, validation_split=0.1, shuffle_option=False)
    print("Training Net...")
    start_time = time.time()
    train_net(net, trainloader, valloader, optimizer, criterion)
    print("Testing Net...")
    recall = test_net(net, testloader)
    end_time = time.time()
    print(f'Training time for batch size 64: {end_time - start_time} seconds')

    print("******************************************")
    print("********  batch size is 256  ********")
    print("******************************************")

    net = Net_5A()
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=1e-3)
    trainloader, testloader, valloader = load_data(batch_size=256, num_workers=4, validation_split=0.1, shuffle_option=False)
    print("Training Net...")
    start_time = time.time()
    train_net(net, trainloader, valloader, optimizer, criterion)
    print("Testing Net...")
    recall_1 = test_net(net, testloader)
    end_time = time.time()
    print(f'Training time for batch size 256: {end_time - start_time} seconds')

    print("******************************************")
    print("********  Training Net with learning rate 0.1 ********")
    print("******************************************")

    net = Net_5A()
    net.to(device)
    optimizer_2 = optim.SGD(net.parameters(), lr=0.1)
    trainloader, testloader, valloader = load_data(batch_size=64, num_workers=4, validation_split=0.1,
                                                   shuffle_option=False)
    print("Training Net...")
    train_net(net, trainloader, valloader, optimizer_2, criterion)
    print("Testing Net...")
    recall_2 = test_net(net, testloader)

    print("******************************************")
    print("********  Training Net with Data shuffling ********")
    print("******************************************")

    net = Net_5A()
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=1e-3)
    trainloader, testloader, valloader = load_data(batch_size=64, num_workers=4, validation_split=0.1,
                                                   shuffle_option=True)
    print("Training Net...")
    train_net(net, trainloader, valloader, optimizer, criterion)
    print("Testing Net...")
    recall_3 = test_net(net, testloader)