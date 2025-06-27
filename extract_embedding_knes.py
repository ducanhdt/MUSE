import os
import pandas as pd
import torch

from src.utils import get_entity_embeddings
from src.dictionary import Dictionary


entity_folder = "/home/cs.aau.dk/qz83md/MUSE/Knes_dataset/data/entity"
seed_alignlinks_folder = "/home/cs.aau.dk/qz83md/MUSE/Knes_dataset/data/seed_alignlinks"

embedding_output_folder = "/home/cs.aau.dk/qz83md/MUSE/data/knes_processed"

embedding_model_map = {'mbert':'bert-base-multilingual-cased',
                        'xlm-r':'FacebookAI/xlm-roberta-base',
                        'me5':'intfloat/multilingual-e5-base'}

if __name__ == "__main__":
    if not os.path.exists(embedding_output_folder):
        os.makedirs(embedding_output_folder)

    entities = {}
    for lang in os.listdir(entity_folder):
        data = pd.read_csv(f"{entity_folder}/{lang}",sep='\t', header=None)
        lang = lang.split('.')[0]
        entities[lang] = data[0].apply(lambda x: x.split("/")[-1]).to_list()

        entity_list = list(set(entities[lang]))
        word2id = {word: i for i, word in enumerate(entity_list)}
        id2word = {i: word for word, i in word2id.items()}
        print(len(id2word), len(word2id))
        print(f"Extracted {len(entity_list)} embeddings for language: {lang}")
        src_dico = Dictionary(id2word, word2id, lang)
        
        _entity_list = [entity.replace('_',' ') for entity in entity_list]
        for model_name in embedding_model_map:
            embeddings = get_entity_embeddings(_entity_list, model_name = embedding_model_map[model_name])    

            src_path = os.path.join(f'{embedding_output_folder}/{lang}_{model_name}.pth')
            print('Writing source embeddings to %s ...' % src_path)
            torch.save({'dico': src_dico, 'vectors': embeddings}, src_path)

            # src_path = os.path.join(f'knes_data/pth/{lang}_{model_name}.tsv')
            # print('Writing source embeddings to %s ...' % src_path)
            
            # with open(src_path, 'w', newline='', encoding='utf-8') as tsvfile:
            #     writer = csv.writer(tsvfile, delimiter='\t')
                
            #     # Write words and embeddings
            #     for word, embedding in zip(entity_list, embeddings):
            #         row = [word] + embedding.tolist()
            #         writer.writerow(row)
            print("Done")

    for link_path in os.listdir(seed_alignlinks_folder):
        src_lang, tgt_lang = link_path.split('.')[0].split('-')
        if not os.path.exists(f'{embedding_output_folder}/{src_lang}_{tgt_lang}'):
            os.makedirs(f'{embedding_output_folder}/{src_lang}_{tgt_lang}')

        # copy embedding files
        for model_name in embedding_model_map:
            src_path = f"{embedding_output_folder}/{src_lang}_{model_name}.pth"
            tgt_path = f"{embedding_output_folder}/{tgt_lang}_{model_name}.pth"
            if os.path.exists(src_path) and os.path.exists(tgt_path):
                os.system(f"cp {src_path} {embedding_output_folder}/{src_lang}_{tgt_lang}/")
                os.system(f"cp {tgt_path} {embedding_output_folder}/{src_lang}_{tgt_lang}/")

        all_pairs = []
        for line in open(f"{seed_alignlinks_folder}/{link_path}"):
            src_idx, tgt_idx = line.strip().split('\t')
            src_idx = int(float(src_idx))
            tgt_idx = int(float(tgt_idx))
            src_entity = entities[src_lang][src_idx]
            tgt_entity = entities[tgt_lang][tgt_idx]
            all_pairs.append(f"{src_entity}\t{tgt_entity}\n")

        with open(f"{embedding_output_folder}/{src_lang}_{tgt_lang}/align.txt", 'w') as f:
            for pair in all_pairs:
                f.write(pair)
        
        
        for train_size in [30,50,70,90]:
            train, test = all_pairs[:int(len(all_pairs) * train_size / 100)], all_pairs[int(len(all_pairs) * train_size / 100):]
            print(f"Train size: {train_size}%, {link_path}")
            with open(f"{embedding_output_folder}/{src_lang}_{tgt_lang}/align_train_{train_size}_{100-train_size}.txt", 'w') as f:
                for pair in train:
                    f.write(pair)
            with open(f"{embedding_output_folder}/{src_lang}_{tgt_lang}/align_test_{train_size}_{100-train_size}.txt", 'w') as f:
                for pair in test:
                    f.write(pair)