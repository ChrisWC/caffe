"""
This class serves to load a definition of a caffe model and run training and testing pipelines
"""
class model:
    def __init__(self, configFile=""):
        self.configFile = configFile

        self.name =""
        self.short_name = ""
        self.model_authors = ""


    def getFunctions(self):
        print "listing Functions/Uses of Datasets"

    def loadConfig(self, modelConfigFile=""):
        print "Loading Config"

        import json
        import os

        with open(modelConfigFile) as modelConfig:
            model = json.load(modelConfig)



    def run(options=[]):
        print "Running Pipeline"
