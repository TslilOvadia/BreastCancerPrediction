# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import sklearn as sk
import seaborn as sb
import DataProcessor as dp

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Matan HaHoMo')
    data_preprocessor = dp.DataProcess()
    data_preprocessor.load_data('train.feats.csv','train.labels.0.csv','train.labels.1.csv')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
