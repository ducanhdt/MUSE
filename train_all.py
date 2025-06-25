import os
import time

lang_pairs = ["el-en", "el-es", "el-fr", "el-ja", "en-fr", "es-en", "es-fr", "ja-en", "ja-es", "ja-fr"]

align_path = "data/knes_data/aligns/"
embedding_path = "data/knes_data/pth/"

def train_all():
    method = 'unsupervised'
    for lang_pair in lang_pairs:
        src_lang, tgt_lang = lang_pair.split('-')
        print("===="*30)
        train_size = 30
        test_path = f"{align_path}/{src_lang}-{tgt_lang}_test_{train_size}_{100-train_size}.txt"
        for embeding_type in ['mbert', 'xlm-r','me5']:
            start = time.time()
            src_emb = f"{embedding_path}/{src_lang}_{embeding_type}.pth"
            tgt_emb = f"{embedding_path}/{tgt_lang}_{embeding_type}.pth"
            command = f"python {method}.py --exp_name {method}_{src_lang}_{tgt_lang}_{embeding_type}_{train_size} --exp_id 0 --src_lang {src_lang} --tgt_lang {tgt_lang} --emb_dim 768 --n_refinement 5 --dico_eval {test_path} --src_emb {src_emb} --tgt_emb {tgt_emb} --export pth --dis_most_frequent 0"
            os.system(command)
            end = time.time()
            log = '\t'.join([method, src_lang, tgt_lang, embeding_type, str(train_size), str(end - start)])
            with open("time.txt", "a") as f:
                f.write(f"{log}\n")

            for train_size in [50, 70, 90]:
                src_emb = f"dumped/unsupervised_{src_lang}_{tgt_lang}_{embeding_type}_30/0/vectors-{src_lang}.pth"
                tgt_emb = f"dumped/unsupervised_{src_lang}_{tgt_lang}_{embeding_type}_30/0/vectors-{tgt_lang}.pth"
                command = f"python evaluate.py --exp_name {method}_{src_lang}_{tgt_lang}_{embeding_type}_{train_size} --exp_id 0 --src_lang {src_lang} --tgt_lang {tgt_lang} --emb_dim 768 --dico_eval {test_path} --src_emb {src_emb} --tgt_emb {tgt_emb}"
                print(command)
                os.system(command)

    method = 'supervised'
    for lang_pair in lang_pairs:
        src_lang, tgt_lang = lang_pair.split('-')
        for train_size in [30, 50, 70, 90]:
            train_path = f"{align_path}/{src_lang}-{tgt_lang}_train_{train_size}_{100-train_size}.txt"
            test_path = f"{align_path}/{src_lang}-{tgt_lang}_test_{train_size}_{100-train_size}.txt"
            for embeding_type in ['mbert', 'xlm-r', 'me5']:
                print("===="*30)
                start = time.time()
                print(lang_pair, train_size, method)
                src_emb = f"{embedding_path}/{src_lang}_{embeding_type}.pth"
                tgt_emb = f"{embedding_path}/{tgt_lang}_{embeding_type}.pth"
                command = f"python {method}.py --exp_name {method}_{src_lang}_{tgt_lang}_{embeding_type}_{train_size} --exp_id 0 --src_lang {src_lang} --tgt_lang {tgt_lang} --emb_dim 768 --n_refinement 5 --dico_train {train_path} --dico_eval {test_path} --src_emb {src_emb} --tgt_emb {tgt_emb} --export pth "
                print(command)
                os.system(command)
                end = time.time()
                log = '\t'.join([method, src_lang, tgt_lang, embeding_type, str(train_size), str(end - start)])
                with open("time.txt", "a") as f:
                    f.write(f"{log}\n")
if __name__ == "__main__":
    with open("time.txt", "w") as f:
        f.write("method\tsrc_lang\ttgt_lang\tembeding_type\ttrain_size\ttime\n")
    train_all()
                