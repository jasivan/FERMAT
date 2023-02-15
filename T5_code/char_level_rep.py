
import torch
from torch.nn import Linear, Tanh, Module
from typing import Tuple, List
from transformers import BartTokenizer, BartTokenizerFast
import re


def char_level_representation_impl(text:list) -> Tuple[List[str], List[tuple]]:    
    '''
        Method for splitting all numbers in a tokenized text which are larger
        than one digit. Also removes the artificial '[F]'
        :param text: a tokenized text
        :return: the same tokenized text but with digit-by-digit tokenization 
            and the indices of splitted numbers in the table
    '''

    # find all indices of '[F]', enclosing the numbers of the table
    # and pair consecutive indices into tuples (e. g. [(91, 95), (101, 105), ...])
    num_indices = [i for i, _ in enumerate(text) if _ == '[F]']
    num_indices = list(zip(num_indices[::2], num_indices[1::2]))

    # only run this code if there were actually tokens splitted
    if len(num_indices) > 0:

        # initialize the splitted text with the part of the table preceding the numbers
        splitted_text = text[:num_indices[0][0]]

        # init the list for the indices of splitted numbers
        splitted_indices = []

        # now do two different things: tokenize numbers digit-by-digit and add the them and text
        # between them to splitted_text
        for i, num_indice in enumerate(num_indices):

            # decompose the span described by num_indice into single characters
            # (+1 to ignore the leading '[F]')
            splitted = list(''.join(text[num_indice[0]+1:num_indice[1]]))        

            # here, the word beginning token (\0120) is concatenated with the first digit to represent
            # the start of a word; if the splitted number consists of more than two characters, 
            # they are added as additional elements
            splitted = [''.join(splitted[:2])] + splitted[2:] if len(splitted) > 2 \
                else [''.join(splitted[:2])]

            # add as indice for splitted number the current lenght of splitted_text as start
            # position and this + the length of splitted as end position
            splitted_indices.append((len(splitted_text), len(splitted_text) + len(splitted)))

            # extend the splitted text with the now splitted number
            splitted_text.extend(splitted)

            if i + 1 < len(num_indices):

                # if there are more numbers to be decomposed, extend the splitted text with the part 
                # between the current number and the next number
                splitted_text.extend(text[num_indice[1] + 1:num_indices[i+1][0]])
            
        # extend the splitted text with the remaining parts of the original tokenized text 
        # (e. g. the caption)
        splitted_text.extend(text[num_indices[-1][1] + 1:])

    else:
        splitted_text = text

    # the strip is necessary to avoid <unk> tokens afterwards
    return [t.strip() for t in splitted_text], splitted_indices if len(num_indices) > 0 else []

def char_level_representation_encoding(text:str, tokenizer:BartTokenizer, max_length:int):
    '''
        method for tokenization and encoding as proposed for Digit
        Tokenization (https://arxiv.org/pdf/2004.04487.pdf)
        padding to max length is automaticall applied
        :param text_list: text to process
        :param tokenizer: tokenizer to use
        :return: Dictionary of input_ids and attentions_masks for the 
            passed batch
    '''
    
    # add special characters for marking the beginning and ending of a
    # sentence.    
    if isinstance(tokenizer, BartTokenizer) or isinstance(tokenizer, 
        BartTokenizerFast):
        text = '<s>' + text + '</s>'
    else:
        text = text + '</s>'

    # mark all numbers with '[F]' for 'float' to find them after tokenization
    # '[F]' was at start added as a special token
    # this regular expressions detects all numbers (whether it is part of a word or not) and surrounds it with
    # [F] [F] (\g<0> describes the whole matching group)
    text = re.sub(r'[+-]?(\d*\.)?\d+', r'[F] \g<0>[F]', text)
    
    # generate input_ids...    
    text, splitted_indices = char_level_representation_impl(tokenizer.tokenize(text))    

    input_id = tokenizer.convert_tokens_to_ids(text)

    input_id, attention_mask = pad_or_trim(torch.LongTensor(input_id), max_length, torch.LongTensor([2]), torch.LongTensor([1]))

    # trim splitted indices
    splitted_indices = [indice for indice in splitted_indices if indice[1] < max_length]

    # logging of some samples
    #rnd_logging(file_logger_dt, {'text': text,
    #    'splitted_indices': splitted_indices, 
    #    'input_id': input_id, 
    #    'attention_mask':attention_mask})
    
    return {'input_ids': input_id.squeeze().tolist(),
        'attention_mask': attention_mask.squeeze().tolist(),
        'splitted_indices': splitted_indices,
        'tokenized': text}

def pad_or_trim(tensor:torch.Tensor, max_length:int, eos:torch.Tensor, pad:torch.Tensor):
    '''
        method for padding or trimming of a passed tensor along the
        first dimension
        :param tensor: tensor to be padded or trimmed
        :param max_length: the max length allowed
        :param pad: the object to use for padding (maybe "[pad]" or 
            already its embedding)
        :return: padded / trimmed tensor and its attention mask
    '''

    attention_mask = torch.ones(max_length)

    if tensor.size()[0] > max_length:
        tensor = torch.narrow(tensor, 0, 0, max_length)
        # set the last value to </s>
        tensor[-1] = eos
    elif tensor.size()[0] == max_length:
        # set the last value to </s>
        tensor[-1] = eos    
    elif tensor.size()[0] < max_length:    
        attention_mask = torch.ones(tensor.size()[0])
        attention_mask = torch.cat((attention_mask, torch.zeros(max_length - tensor.size()[0])))
        tensor = torch.cat((tensor, torch.cat([pad] * (max_length - tensor.size()[0]))))        
    
    if len(attention_mask.size()) == 2:        
        attention_mask = torch.max(attention_mask, 1)[0]

    return tensor, attention_mask            