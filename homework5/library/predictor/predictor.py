from allennlp.data.dataset_readers import DatasetReader
from allennlp.models import Model
from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

from typing import List


@Predictor.register("sentence_classifier_predictor")
class SentenceClassifierPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=True)

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence" : sentence})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.split_words(sentence)
        return self._dataset_reader.text_to_instance([str(t) for t in tokens])
    