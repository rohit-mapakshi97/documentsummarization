#!/usr/bin/python3
class Summarizer: 
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.model = self.loadModel(model_name)

    def loadModel(self, model_name): 
        # Load the appropriate model from the "./models/" directory 
        pass

    def summarize(sel, input_text): # [TODO] add return types here: summary, headline and metrics 
        # 1. Preprocess the input_text 
        # 2. Generate summary and meterics form the selected model 
        pass