from utils import create_data_lists

if __name__ == "__main__":
    # create_data_lists(train_folders = ["../test2015"], val_folders = ["../val2017"], test_folders = ["../Set5", "../Set14", "../BSDS100"], min_size = 100, output_folder = "./")
    create_data_lists(train_folders = ["../DIV2K_train_HR"], val_folders = ["../test"], test_folders = ["../DIV2K_valid_HR"], min_size = 100, output_folder = "./")
