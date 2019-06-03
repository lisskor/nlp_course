import os
import unicodedata
import string

from typing import Iterator, List, Dict

from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.token_indexers.token_characters_indexer import TokenCharactersIndexer
from allennlp.data.tokenizers import Token

@DatasetReader.register('names-reader')
class NamesDatasetReader(DatasetReader):
    """
    DatasetReader for names data
    
    Each file, its name specifying the language,
    contains one last name per line.
    
    read() method accepts the path to the directory
    containing all language files
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None,
                 token_character_indexers: Dict[str, TokenCharactersIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        # add indexer for characters
        self.token_character_indexers = token_character_indexers or {"token_characters": TokenCharactersIndexer()}
        # following the PyTorch tutorial, turn everything to plain ASCII
        self.all_letters = string.ascii_letters + " .,;'"
        
    def text_to_instance(self, name: List[Token], label: str = None) -> Instance:
        name_field = TextField(name, self.token_indexers)
        fields = {"name": name_field}
        
        fields["name_characters"] = TextField(name, self.token_character_indexers)

        if label:
            label_field = LabelField(label=label)
            fields["label"] = label_field

        return Instance(fields)
    
    def _read(self, files_dir: str) -> Iterator[Instance]:
        # first, get all languages from filenames
        lang_files = [f for f in os.listdir(files_dir)
                      if os.path.isfile(os.path.join(files_dir, f))]
        # iterate over files
        for file in lang_files:
            # file_path is full path to file to open it
            file_path = os.path.join(files_dir, file)
            # file_label is the language name
            file_label = os.path.splitext(file)[0]
            with open(file_path) as f:
                for line in f:
                    # following the PyTorch tutorial, turn everything to plain ASCII
                    name = ''.join(c for c in unicodedata.normalize('NFD', line.strip())
                                   if unicodedata.category(c) != 'Mn'
                                   and c in self.all_letters)
                    yield self.text_to_instance([Token(name)], file_label)