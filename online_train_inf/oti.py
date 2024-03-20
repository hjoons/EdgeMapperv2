import threading
import queue
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import time
from torch import nn
import torch.nn.functional as F
import os

class Net(nn.Module):
    def __init__(self, num_classes=10, channels=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool3(F.relu(self.conv5(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def reload_weights(self, weights_path="net.pth"):
        # Load the weights from the given file path
        state_dict = torch.load(weights_path)
        # Update the current model state with the loaded state dictionary
        self.load_state_dict(state_dict)
        print("Model weights reloaded successfully.")

    def save_weights(self, weights_path="net.pth"):
        # Get the model's current state dictionary
        state_dict = self.state_dict()
        # Save the state dictionary to the specified path
        torch.save(state_dict, weights_path)
        print("Model weights saved successfully.")


def inference_thread():
    fps = 24
    time_per_frame = 1 / fps  # 0.041666666666666664 = 41.6ms
    # CIFAR10 class labels
    model_path = "net.pth"
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    modelI = Net().to(device)
    modelI.eval()
    modelI.reload_weights()
    last_mod_time = os.path.getmtime(model_path)
    while True:
        for i in range(len(trainset)):
            start_time_all = time.time()
            if stop_event.is_set():
                break
            if os.path.getmtime(model_path) != last_mod_time:
                modelI.reload_weights()
                last_mod_time = os.path.getmtime(model_path)
            # Load one sample at a time
            image, label = trainset[i]
            # Put the (image, label) tuple in the buffer
            if buffer.qsize() >= max_buffer_size:
                # Remove the oldest item to make space for the new one
                buffer.get()  # This line removes the oldest entry from the queue
                print("Removed the oldest entry from the buffer.")
            buffer.put((image.cpu(), label))  # Move image back to CPU before storing in buffer
            print("Added a new entry to the buffer.")
            image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device

            with torch.no_grad():
                outputs = modelI(image)  # Perform inference
                _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
                predicted_class = classes[predicted[0].item()]

            # print(f'[{i}] Predicted: "{predicted_class}", Actual: "{classes[label]}"')


            time.sleep(max(0., time_per_frame - (time.time() - start_time_all)))
            # if buffer.qsize() >= max_buffer_size:
            #     time.sleep(1)  # Wait a bit if the buffer is full, to give the training thread time to consume


def training_thread():
    modelT = Net().to(device)
    modelT.train()
    modelT.reload_weights()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(modelT.parameters(), lr=0.001, momentum=0.9)
    # while not stop_event.is_set() or not buffer.empty():
    while True:
        if buffer.qsize() >= max_buffer_size:
            # Create a dataset from the buffer for training
            dataset = [buffer.get() for _ in range(max_buffer_size)]
            images, labels = zip(*dataset)
            images = torch.stack(images).to(device)
            labels = torch.tensor(labels).to(device)

            # Training step
            optimizer.zero_grad()
            outputs = modelT(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print(f"Trained on a batch of {len(images)} images. Loss: {loss.item()}")
            modelT.save_weights("net.pth")
        else:
            time.sleep(1)  # Wait for the buffer to fill

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR10 Data loading and normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Assuming 'Net' is your PyTorch model
modelG = Net().to(device)
modelG.train()  # Make sure the model is in training mode
modelG.save_weights()
# Criterion and optimizer


# The buffer to store data for training
buffer = queue.Queue()
max_buffer_size = 128  # Define the maximum size of the buffer

# Stop event to cleanly exit threads
stop_event = threading.Event()
def run_oit():
    # Starting the threads
    inference = threading.Thread(target=inference_thread)
    training = threading.Thread(target=training_thread)

    inference.start()
    training.start()

    # Running for a limited time or until a certain condition, then stopping
    # try:
    #     inference.join(timeout=10)  # Let's say we run this for 10 seconds for demonstration
    #     training.join(timeout=10)
    # except KeyboardInterrupt:
    #     pass
    #
    # stop_event.set()
    #
    # inference.join()
    # training.join()
    # return None

if __name__ == '__main__':
    run_oit()