import logging
import pickle as pk

log_name = "log.txt"
logging.basicConfig(level=logging.INFO, 
    filename=log_name, 
    filemode='a', 
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s') 

class ResultRecorder:
    def __init__(self):
        self.result = {}
        logging.info("=========================================================")
        logging.info("====================== Begin  ===========================")
        logging.info("=========================================================\n")
        pass

    def print(self, *argv):
        info = ""
        for data in argv:
            info += str(data)
        print(info)
        logging.info(info)

    def print_result(self, *argv):
        info = ""
        for data in argv:
            info += "#Result{%s}" % str(data)
        print(info)
        logging.info(info)
        
    def store(self, *argv):
        for data in argv:
            if data.find(":") < 0:
                continue
            label, data = data.split(":")
            self.store_pair(label, data)

    def write_result(self, *argv):
        self.print_result(*argv)
        self.store(*argv)

    def store_pair(self, label, data):
        if label not in self.result:
            self.result[label] = []
        self.result[label].append(data)

    def write_pair(self, label, data):
        self.print_result(label + ":" + str(data))
        self.store_pair(label, data)

    def dump(self, f):
        pk.dump(self.result, f)

