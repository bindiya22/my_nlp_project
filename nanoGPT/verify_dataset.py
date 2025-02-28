import os
import numpy as np

# Paths to the generated files (updated to new directory and file names)
train_txt_path = "C:\Users\Bindiya\Documents\NLProject\nanoGPT\data\my_dataset\train_new.txt"
val_txt_path = "C:\Users\Bindiya\Documents\NLProject\nanoGPT\data\my_dataset\val_new.txt"
train_bin_path = "C:\Users\Bindiya\Documents\NLProject\nanoGPT\data\my_dataset\train_new.bin"
val_bin_path = "C:\Users\Bindiya\Documents\NLProject\nanoGPT\data\my_dataset\val_new.bin"

# Verify sizes of the binary files
def verify_file_sizes():
    train_size = os.path.getsize(train_bin_path)
    val_size = os.path.getsize(val_bin_path)
    print(f"Train.bin size: {train_size} bytes")
    print(f"Val.bin size: {val_size} bytes")

# Display first few lines from text files
def view_text_files():
    print("\nFirst 5 lines from train_new.txt:")
    with open(train_txt_path, 'r', encoding='utf-8') as train_file:
        for _ in range(5):
            print(train_file.readline().strip())

    print("\nFirst 5 lines from val_new.txt:")
    with open(val_txt_path, 'r', encoding='utf-8') as val_file:
        for _ in range(5):
            print(val_file.readline().strip())

# Display first few tokens from binary files
def view_binary_files():
    print("\nFirst 10 tokens from train_new.bin:")
    train_data = np.memmap(train_bin_path, dtype=np.uint16, mode='r')
    print(train_data[:10])

    print("\nFirst 10 tokens from val_new.bin:")
    val_data = np.memmap(val_bin_path, dtype=np.uint16, mode='r')
    print(val_data[:10])

# Execute the verification steps
if __name__ == "__main__":
    print("Verifying file sizes...")
    verify_file_sizes()

    print("\nViewing contents of text files...")
    view_text_files()

    print("\nViewing contents of binary files...")
    view_binary_files()
