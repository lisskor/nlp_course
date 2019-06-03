from typing import Dict

import torch

from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer

@Model.register('names-classifier')
class NamesClassifier(Model):
    
    def __init__(self,
                 char_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.char_embeddings = char_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.accuracy = CategoricalAccuracy()
        
    def forward(self,
                name: Dict[str, torch.Tensor],
                name_characters: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        
        mask = get_text_field_mask(name_characters)
        embeddings = self.char_embeddings(name_characters)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        
        if label is not None:
            self.accuracy(tag_logits, label)
            output["loss"] = self.loss_function(tag_logits, label)

        return output
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}