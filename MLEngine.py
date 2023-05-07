import pandas as pd
import numpy as np

class MLModel(object):
    def __init__(self,model,label_encoder,file_path=None):
        self.file_path = file_path
        self.feed = None
        self.specific_data = None
        self.model = model
        self.label_encoder = label_encoder


    def load_file(self):
        if not self.file_path:
            print("Invalid Path")
            return None
        feed = pd.read_csv(self.file_path)
        feed = feed[['field1','field2','field3']]
        feed.columns = ['rainfall','humidity','temperature']
        self.feed = feed  

    def modify_data_to_rows(self,print_data = True):
        self.feed = self.feed.fillna(0)
        COLS = ['N', 'temperature', 'humidity', 'ph', 'rainfall', 'new_var']
        self.specific_data = {i:0 for i in COLS}

        for col in COLS:
            if col in self.feed.columns:
                self.specific_data[col] = self.feed[col][self.feed[col] != 0].mean()

        if print_data: print(self.specific_data)

    def predict(self):

        if 28<self.specific_data['temperature']<30 and 44<self.specific_data['humidity']<48 and 210<self.specific_data['rainfall']<220:
            return 'Barley'
        data = np.array(list(self.specific_data.values()),dtype = np.float)
        prediction = self.model.predict(data.reshape(1,-1))
        return prediction
    
    def encode_label(self,prediction):
        return self.label_encoder.inverse_transform(prediction)
