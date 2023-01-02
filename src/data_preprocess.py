import pandas as pd
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv('data/diabetic_data.csv')
    print(data.info())
    data.describe().to_csv('data_info.csv')
    # for name in data.columns:
    #     print(name)
    #     col = data[name]
        
    
if __name__ == '__main__':
    main()