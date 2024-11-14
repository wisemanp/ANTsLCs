import os
import zipfile

file_path = "C:/Users/laure/OneDrive/Desktop/YoRiS/September/other ANT data/ASASSN-18jd_lc_files/ASASSN-18jd_paper_lc_file.zip"
folder_directory = "C:/Users/laure/OneDrive/Desktop/YoRiS/September/other ANT data/ASASSN-18jd_lc_files/"

with zipfile.ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extractall(folder_directory)


