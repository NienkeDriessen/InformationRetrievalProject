import pandas as pd
import os

DATA_FOLDER = "../data_openImages"
METADATA = "../ellende4.csv"

"""
File structure:

p2p/OpenImages
- OJ -> folders in path + id of image + extension
- SDv4 -> folders in path + path stored in metadata
- SDv5 -> folders in path + id of image + extension
original/images -> id + extension
"""

# # for each row in the metadata, check if the image exists with a path according to the comment above
metadata = pd.read_csv(METADATA)
mdata = pd.DataFrame(columns=metadata.columns)
rows = []
drop_rows = []
correct_guess = 0
incorrect_guess = 0
semi_correct_guess = 0

for index, row in metadata.iterrows():
    # Construct the expected file path based on the metadata
    folders, filename = os.path.split(row['image_path'])
    name, ext = os.path.splitext(filename)
    # print(folders)
    if "SDv4" in folders:
        expected_path = os.path.join(DATA_FOLDER, row['image_path'])
        ex_folder, ex_filename = folders, name
    elif "SDv5" in folders:
        expected_path = os.path.join(DATA_FOLDER, os.path.join(folders, (row['image_id'])))
        ex_folder, ex_filename = folders, row['image_id']
    elif "OJ" in folders:
        expected_path = os.path.join(DATA_FOLDER, os.path.join(folders, (row['image_id'])))
        ex_folder, ex_filename = folders, row['image_id']

    else:
        expected_path = os.path.join(DATA_FOLDER, row['image_path'])
        ex_folder, ex_filename = folders, name

    # Check if the file exists
    if not os.path.exists(expected_path):
        which_ext = [os.path.exists(os.path.join(DATA_FOLDER, os.path.join(ex_folder, ex_filename) + e )) for e in ['.jpg', '.jpeg', '.png']]
        if any(which_ext):
            # print(f"File exists with different extension: {expected_path}")
            # new_path = path with extension that has true above
            for i in range(3):
                if which_ext[i]:
                    new_path = os.path.join(DATA_FOLDER, os.path.join(ex_folder, ex_filename) + ['.jpg', '.jpeg', '.png'][i])
                    print(f"New path: {new_path}")
                    row['image_path'] = new_path
                    rows.append(row)
                    # mdata.iloc[index] = row
            semi_correct_guess += 1
        elif any([os.path.exists(os.path.join('Altered_Kansinsky_2_2/Kandinsky_2_2', ex_filename) + e) for e in ['.jpg', '.jpeg', '.png']]):
            # print(f"File exists in altered folder: {expected_path} with an extension")
            for i in range(3):
                if which_ext[i]:
                    new_path = os.path.join('Altered_Kansinsky_2_2/Kandinsky_2_2', ex_filename) + ['.jpg', '.jpeg', '.png'][i]
                    row['image_path'] = new_path
                    rows.append(row)

                    # mdata.iloc[index] = row
            semi_correct_guess += 1
        else:
            # print(f"Missing file: {expected_path}")
            metadata.drop(index, axis = 0, inplace=True)
            incorrect_guess += 1
    else:
        # print(f"File exists: {expected_path}")
        # mdata.iloc[index] = row
        correct_guess += 1
        rows.append(row)

print(f"Correct guesses: {correct_guess}")
print(f"Incorrect guesses: {incorrect_guess}")

# mdata = metadata.drop(drop_rows, axis='rows', inplace=False)
# print(rows)
mdata = pd.DataFrame(rows)

for _, row in mdata.iterrows():
    if not os.path.exists(os.path.join(DATA_FOLDER, row['image_path'])):
        raise Exception('altering did not work')

mdata.to_csv('ellende69.csv', index=False)