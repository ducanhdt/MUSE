import os
import pandas as pd
import torch

from src.utils import get_entity_embeddings
from src.dictionary import Dictionary


entity_folder = "/home/cs.aau.dk/qz83md/MUSE/data/dbp15k"

embedding_output_folder = "/home/cs.aau.dk/qz83md/MUSE/data/dbp15k_processed/"

embedding_model_map = {'mbert':'bert-base-multilingual-cased',
                        'xlm-r':'FacebookAI/xlm-roberta-base',
                        'me5':'intfloat/multilingual-e5-base'}

if __name__ == "__main__":
    if not os.path.exists(embedding_output_folder):
        os.makedirs(embedding_output_folder)

    for lang_pair in ['fr_en', 'zh_en', 'ja_en']:
        
        if not os.path.exists(f'{embedding_output_folder}/{lang_pair}'):
            os.makedirs(f'{embedding_output_folder}/{lang_pair}')
        
        source_lang, target_lang = lang_pair.split('_')
        # extract embedding
        data_path = f"{entity_folder}/{lang_pair}/ent_ILLs"
        data = pd.read_csv(data_path, sep='\t', header=None)

        source_entity = data[0].apply(lambda x: x.split("/")[-1]).to_list()
        target_entity =  data[1].apply(lambda x: x.split("/")[-1]).to_list()

        source_entity_list = list(set(source_entity))
        target_entity_list = list(set(target_entity))

        source_word2id = {word: i for i, word in enumerate(source_entity_list)}
        target_word2id = {word: i for i, word in enumerate(target_entity_list)}
        source_id2word = {i: word for word, i in source_word2id.items()}
        target_id2word = {i: word for word, i in target_word2id.items()}

        print(len(source_id2word), len(source_word2id))
        print(f"Extracted {len(source_entity_list)} embeddings for language: {source_lang}")
        print(len(target_id2word), len(target_word2id))
        print(f"Extracted {len(target_entity_list)} embeddings for language: {target_lang}")
        source_dico = Dictionary(source_id2word, source_word2id, source_lang)
        target_dico = Dictionary(target_id2word, target_word2id, target_lang)  
        _source_entity_list = [entity.replace('_',' ') for entity in source_entity_list]
        _target_entity_list = [entity.replace('_',' ') for entity in target_entity_list]

        for model_name in embedding_model_map:
            source_embeddings = get_entity_embeddings(_source_entity_list, model_name = embedding_model_map[model_name])    
            target_embeddings = get_entity_embeddings(_target_entity_list, model_name = embedding_model_map[model_name])    

            src_path = os.path.join(f'{embedding_output_folder}/{lang_pair}/{source_lang}_{model_name}.pth')
            print('Writing source embeddings to %s ...' % src_path)
            torch.save({'dico': source_dico, 'vectors': source_embeddings}, src_path)

            tgt_path = os.path.join(f'{embedding_output_folder}/{lang_pair}/{target_lang}_{model_name}.pth')
            print('Writing target embeddings to %s ...' % tgt_path)
            torch.save({'dico': target_dico, 'vectors': target_embeddings}, tgt_path)


        all_pairs = []
        for src_entity, tgt_entity in zip(source_entity, target_entity):
            all_pairs.append(f"{src_entity}\t{tgt_entity}\n")

        with open(f'{embedding_output_folder}/{lang_pair}/align.txt', 'w') as f:
            for pair in all_pairs:
                f.write(pair)
        
        
        for train_size in [30,50,70,90]:
            train, test = all_pairs[:int(len(all_pairs) * train_size / 100)], all_pairs[int(len(all_pairs) * train_size / 100):]
            print(f"Train size: {train_size}")
            with open(f'{embedding_output_folder}/{lang_pair}/align_train_{train_size}_{100-train_size}.txt', 'w') as f:
                for pair in train:
                    f.write(pair)
            with open(f'{embedding_output_folder}/{lang_pair}/align_test_{train_size}_{100-train_size}.txt', 'w') as f:
                for pair in test:
                    f.write(pair)