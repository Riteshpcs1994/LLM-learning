from sklearn.model_selection import train_test_split
import numpy as np

def stratified_split(texts, l1_labels, l2_labels, test_size=0.2, val_size=0.1, random_state=42):
    """
    Stratified split maintaining L1-L2 relationship distributions
    Returns: (train_texts, val_texts, test_texts,
              train_l1, val_l1, test_l1,
              train_l2, val_l2, test_l2)
    """
    # Create composite labels for stratification
    composite_labels = [f"{l1}_{l2}" for l1, l2 in zip(l1_labels, l2_labels)]
    
    # First split: train+val vs test
    train_val_texts, test_texts, train_val_l1, test_l1, train_val_l2, test_l2 = train_test_split(
        texts, l1_labels, l2_labels,
        test_size=test_size,
        stratify=composite_labels,
        random_state=random_state
    )
    
    # Second split: train vs val
    train_texts, val_texts, train_l1, val_l1, train_l2, val_l2 = train_test_split(
        train_val_texts, train_val_l1, train_val_l2,
        test_size=val_size/(1-test_size),
        stratify=[f"{l1}_{l2}" for l1, l2 in zip(train_val_l1, train_val_l2)],
        random_state=random_state
    )
    
    return (train_texts, val_texts, test_texts,
            train_l1, val_l1, test_l1,
            train_l2, val_l2, test_l2)

# Usage example
if __name__ == "__main__":
    # Sample data (replace with your dataset)
    texts = [
        "How do I reset my password?", 
        "I want to upgrade my plan",
        "Where is my order?",
        "Can I cancel my purchase?",
        "Why was my payment declined?",
        "How to change my email",
        "Need premium features",
        "Track package status",
        "Return item request",
        "Payment method issues"
    ]
    l1_labels = [0, 1, 2, 2, 0, 0, 1, 2, 2, 0]  # 0=Account, 1=Subscription, 2=Orders
    l2_labels = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]  # 0=Password, 1=Upgrade, 2=Tracking, 3=Cancel, 4=Payment
    
    # Perform stratified split
    (train_texts, val_texts, test_texts,
     train_l1, val_l1, test_l1,
     train_l2, val_l2, test_l2) = stratified_split(texts, l1_labels, l2_labels)
    
    # Create datasets
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    train_dataset = ConversationDataset(train_texts, train_l1, train_l2, tokenizer)
    val_dataset = ConversationDataset(val_texts, val_l1, val_l2, tokenizer)
    test_dataset = ConversationDataset(test_texts, test_l1, test_l2, tokenizer)
    
    # Verify distributions
    def print_distribution(name, l1, l2):
        print(f"\n{name} Distribution:")
        print(f"L1: {np.unique(l1, return_counts=True)}")
        print(f"L2: {np.unique(l2, return_counts=True)}")
    
    print_distribution("Train", train_l1, train_l2)
    print_distribution("Validation", val_l1, val_l2)
    print_distribution("Test", test_l1, test_l2)



    ########################################################


    import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

# 1. Create complete dataset first
full_dataset = ConversationDataset(
    texts=all_texts,
    l1_labels=all_l1_labels,  # List[int] 
    l2_labels=all_l2_labels,  # List[int]
    tokenizer=tokenizer
)

# 2. Create composite labels for stratification
def get_composite_labels(dataset):
    """Extract L1-L2 combinations from dataset"""
    return [
        f"{sample['l1_labels'].item()}_{sample['l2_labels'].item()}"
        for sample in dataset
    ]

# 3. Perform stratified split
def stratified_dataset_split(dataset, test_size=0.2, val_size=0.1, random_state=42):
    indices = np.arange(len(dataset))
    composite_labels = get_composite_labels(dataset)
    
    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=composite_labels,
        random_state=random_state
    )
    
    # Second split: train vs val
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size/(1-test_size),
        stratify=[composite_labels[i] for i in train_val_idx],
        random_state=random_state
    )
    
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)

# 4. Split the dataset
train_dataset, val_dataset, test_dataset = stratified_dataset_split(full_dataset)

# 5. Verify distributions (optional)
def print_distribution(subset):
    l1_counts = {}
    l2_counts = {}
    for sample in subset:
        l1 = sample['l1_labels'].item()
        l2 = sample['l2_labels'].item()
        l1_counts[l1] = l1_counts.get(l1, 0) + 1
        l2_counts[l2] = l2_counts.get(l2, 0) + 1
    print("L1 Distribution:", l1_counts)
    print("L2 Distribution:", l2_counts)

print("Training Set:")
print_distribution(train_dataset)
print("\nValidation Set:")
print_distribution(val_dataset)
print("\nTest Set:")
print_distribution(test_dataset)


train_ds, val_ds, test_ds = stratified_dataset_split(full_dataset)

# Create DataLoaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)
test_loader = DataLoader(test_ds, batch_size=64)