from transformers import RobertaTokenizer

from DatasetRepresentation.DataPreprocessing.TextProcessing import article_column_name

roberta_column_name="roberta_column_name"
def generate_roberta_embeddings_for_training_set(train_df,text_name=None):
    if text_name==None:
        text_name=article_column_name
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    roberta_input = []
    for sentense in train_df[text_name].values:
        current = tokenizer.encode_plus(
            sentense,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            truncation=True,
            max_length=None ,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        roberta_input.append(current)
    return roberta_input

def get_roberta_embeddings_for_training_set(train_df,hyperparameters):
    return train_df[roberta_column_name].tolist()
