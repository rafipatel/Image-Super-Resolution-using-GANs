import tensorflow_datasets as tfds

div2k_data = tfds.image.Div2k(config="bicubic_x4")
div2k_data.download_and_prepare()

