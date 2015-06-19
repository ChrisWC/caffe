class dataset:
    def __init__(self, dataset_config="dataset.config"):
        self.dataset_config = dataset_config
    def loadConfig(self, dataset_config, mergeConfig):
        print "Loading Dataset Configuration"

    def download(self):
        print "Downloading Dataset"

    def getDirectories(self, withFormats=[], allowOtherFormats=True):
        print "Listing Directories with format: "

    def getFiles(self, directories=[], withFormats=[], withFunctions=[]):
        print "Getting Files with formats..."
