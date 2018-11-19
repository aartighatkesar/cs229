class Model(object):
    #abstract model for tensorflow

    def add_placeholders(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def build_graph(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_training_step(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def run_train_iter(self,inputs_batch,labels_batch):
        raise NotImplementedError("Each Model must re-implement this method.")

    def get_validation_accuracy(self,sess):
        raise NotImplementedError("Each Model must re-implement this method.")

    def get_test_data(self, sess):
        raise NotImplementedError("Each Model must re-implement this method.")

    def run_epoch(self, sess):
        raise NotImplementedError("Each Model must re-implement this method.")

class NeuralLayer(object):
    def build_graph(self,values):
        raise NotImplementedError("Each Model must re-implement this method.")