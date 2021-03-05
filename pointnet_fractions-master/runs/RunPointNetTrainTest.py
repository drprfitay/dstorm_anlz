from models.Classifiers import PointNetClassifier
from trainers.ClassificationTrainer import ClassificationTrainer

class RunPointNetClassifierNAS:

    def __init__(self, setup):

        self._setup = setup
        self.print_to_log = self._setup.print_to_log



    def __call__(self, name):

        classifier = PointNetClassifier(self._setup)
        trainer = ClassificationTrainer(self._setup)



