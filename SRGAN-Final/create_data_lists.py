from utils import create_data_lists

if __name__ == "__main__":
    create_data_lists(train_folders = ["../DIV2K_train_HR"], val_folders = ["../Set14"], test_folders = ["../BSDS100", "../DIV2K_valid_HR"], min_size = 100, output_folder = "./")