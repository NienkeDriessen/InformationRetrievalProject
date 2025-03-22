import tarfile

file_path = "../data_tar_files/bigOpenImages1.tar.bz2"
extract_path = "../data_p2p"

with tarfile.open(file_path, "r:bz2") as tar:
    tar.extractall(extract_path)
    print("Extraction completed!")
