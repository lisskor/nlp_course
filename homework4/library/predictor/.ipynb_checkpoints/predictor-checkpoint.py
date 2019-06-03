import torch
import torch.optim as optim

import numpy as np

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.util import JsonDict
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.predictors import Predictor


@Predictor.register('names-predictor')
class NamesPredictor(Predictor):
    
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        return label_dict[np.argmax(output_dict['tag_logits'])]
    
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        name = [Token(json_dict['name'])]
        return self._dataset_reader.text_to_instance(name=name)