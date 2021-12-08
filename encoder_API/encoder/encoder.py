from transformers import AutoModel, AutoTokenizer
import pandas as pd
import torch

"""
Adapted from code from the following article: https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
"""


class encoder:
    """
    Raised if size of a padded list of tokens contains 512 or more elements
    """

    class SizeError(Exception):
        pass

    """
    This is the __init__ for the encoder class.

    Parameters
    ----------
    config : dict
        Dictionary with 'modelName' and 'tokenizerName' being str links to HF transformers BERT repos
    """

    def __init__(self, config):
        self.model = AutoModel.from_pretrained(config['modelName'], output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(config['tokenizerName'])

        self.model.eval()

    """
    Converts a tokenized sent into BERT-readable PyTorch tensors.
    
    This is specifically to be used in tandem with get_bert_embeddings().
    
    Parameters
    ----------
    tokenized_sent : list
        This should be a Python list() object of the tokenized text. 
    
    Returns
    ------- 
    tokens : list
        Tokens (padded if needed)
    tokens_tensor : torch.Tensor
        Token IDs
    segments_tensors : torch.Tensor
        Segments IDs
    """

    def preprocess(self, tokenized_sent):
        # make sure tokenized_sent is properly padded!
        padSent = lambda sent: sent if (sent[0] == '[CLS]' and sent[-1] == '[SEP]') else ['[CLS]'] + sent + ['[SEP]']
        tokens = padSent(tokenized_sent)

        # create token indices and segment IDs
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        segments_ids = [1] * len(indexed_tokens)

        # convert to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        return tokens, tokens_tensor, segments_tensors

    """
    Embeds a list of tokens into contextualized word embeddings.
    
    Not to be confused with embed(), its wrapper method
    
    Parameters
    ----------
    tokenized_text : list
        padded tokenized text
    tokens_tensor : torch.Tensor
        PyTorch tensor containing token IDs
    segments_tensor : torch.Tensor
        PyTorch tensor containing segments IDs
    asDF : bool
        Dictates whether the method returns embeddings in the form of a dictionary or Pandas DataFrame
    
    Returns
    -------
    dict
        Returns tokens and embeddings ('tokens', 'embeddings', respectively) in dictionary (or Pandas DataFrame, if specified)
    """

    def get_bert_embeddings(self, tokenized_text, tokens_tensor, segments_tensor, asDF=False):
        if len(tokenized_text) < 512:
            with torch.no_grad():
                output = self.model(tokens_tensor, segments_tensor)
                hidden_states = output[2][1:]

            token_embeddings = hidden_states[-1]
            token_embeddings_squeezed = torch.squeeze(token_embeddings, dim=0)

            # prepare embeddings, tokens for export
            data = dict()
            data['tokens'] = tokenized_text
            # have to put into list to make JSON-friendly :(
            data['embeddings'] = [embed.tolist() for embed in token_embeddings_squeezed]

            if asDF:
                return pd.DataFrame(data)
            else:
                return data

        else:
            raise encoder.SizeError('Text is too large!')

    """
    Wrapper method for get_bert_embeddings() and preprocess()
    
    Parameters
    ----------
    tokenized_text : list
        List of tokens
    
    Returns
    -------
    embeddings : pd.DataFrame
        Embeddings in a dataframe (`columns= ['tokens', 'embeddings']`)
    """

    def embed(self, tokenized_text, deliminators=None):
        tokens, tokens_tensor, segments_tensor = self.preprocess(tokenized_text)
        try:
            embeddings = self.get_bert_embeddings(tokens, tokens_tensor, segments_tensor, asDF=True)
            msg = 'successfully embedded text:\n' + ' '.join(tokenized_text)
            return embeddings, msg

        except encoder.SizeError:
            delims = [':'] if deliminators is None else deliminators
            broken_up_sents = [[]]
            for i in tokenized_text:
                broken_up_sents[-1].append(i)
                if any(delim in i for delim in delims):
                    broken_up_sents.append([])

            if all(len(sent) < 512 for sent in broken_up_sents):
                data = {'tokens': [], 'embeddings': []}
                for sent in broken_up_sents:
                    tokens, tokens_tensor, segments_tensor = self.preprocess(sent)
                    embeddings = self.get_bert_embeddings(tokens, tokens_tensor, segments_tensor, asDF=False)
                    data['tokens'] += embeddings['tokens']
                    data['embeddings'] += embeddings['embeddings']

                msg = 'successfully embedded text:\n' + ' '.join(sent)
                return pd.DataFrame(data), msg

            else:
                msg = 'failed to embed text:\n' + ' '.join(tokenized_text)
                print(msg)
                return pd.DataFrame({'tokens': list(), 'embeddings': list()}), msg
