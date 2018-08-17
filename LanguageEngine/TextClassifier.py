class Classifier():
    """
    A class for initializing the specified classifier

    Attributes
    ----------
    SIMPLE_CLASSIFIER : int
        The integer denotation of the SimpleClassifer, used to instantiate classifer
    classifier : LanguageEngine.model
        The class object of the classifier specified
    
    Methods
    -------
    predict(sentence)
        returns the dict containing the original sentenece, predicted class, and the slots (entities)
    """

    SIMPLE_CLASSIFIER = 1

    def __init__(self, type=Classifier.SIMPLE_CLASSIFIER):
        if type is Classifier.SIMPLE_CLASSIFIER:
            from LanguageEngine.models.SimpleClassifier import SimpleClassifier as sc
            self.classifier = sc
    
    def predict(self, sentence):
        """Does prediction on the given sentence
        Predicts the class of the sentece and the slots (entities)

        Parameters
        ----------
        sentence : str
            The input sentence that needs to be classified
        Returns
        -------
        returnJson : dict
            The dict containing the class, original sentence and the predicted entities of the sentence
        """
        returnJson = self.classifier.predict(sentence)
        return returnJson