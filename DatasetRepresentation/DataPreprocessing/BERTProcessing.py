from transformers import RobertaTokenizer

def generate_roberta_embeddings_for_training_set(train_df,
                                                 article_column_name="article"):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    roberta_input = []
    for sentense in train_df[article_column_name].values:
        current = tokenizer.encode_plus(
            sentense,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            truncation=True,
            max_length=10,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        roberta_input.append(current)
    return roberta_input

def get_roberta_embeddings_for_training_set(train_df,hyperparameters):
    roberta_column_name= hyperparameters["roberta_column_name"]
    return train_df[roberta_column_name].tolist()
