'''This is the repo which contains the original code to the WACV 2021 paper
"Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows"
by Marco Rudolph, Bastian Wandt and Bodo Rosenhahn.
For further information contact Marco Rudolph (rudolph@tnt.uni-hannover.de)'''

import config as c
from train import train
from utils import load_datasets, make_dataloaders
import time

def train_and_time(class_name):
    train_set, test_set = load_datasets(c.dataset_path, class_name)
    train_loader, test_loader = make_dataloaders(train_set, test_set)
    
    # Set model name for the current class
    c.modelname = f"DAGM_model_{class_name}"
    
    start_time = time.time()
    model = train(train_loader, test_loader)
    end_time = time.time()
    
    training_time = end_time - start_time
    print(f"Training time for {class_name}: {training_time:.2f} seconds")
    
    return model, training_time

# Train and time each class
results = {}
for class_name in c.class_names:
    print(f"\nTraining model for {class_name}")
    model, training_time = train_and_time(class_name)
    results[class_name] = {'model': model, 'time': training_time, 'model_name': c.modelname}

# Print summary of training times and model names
print("\nTraining Summary:")
for class_name, result in results.items():
    print(f"{class_name}:")
    print(f"  Model name: {result['model_name']}")
    print(f"  Training time: {result['time']:.2f} seconds")

# Calculate and print total training time
total_time = sum(result['time'] for result in results.values())
print(f"\nTotal training time for all classes: {total_time:.2f} seconds")