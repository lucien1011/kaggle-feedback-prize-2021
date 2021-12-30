preprocess_conf = dict(
    input_train_csv_path='storage/train.csv',
    input_train_text_dir='storage/train/',
    base_dir='storage/output/211228_baseline/',
    seed=42,
    type='sent_classification',
)

train_conf = dict(
    num_class=8,
    discourse_df_path='storage/train.csv',
    base_dir='storage/output/211228_baseline/',
    rev_dir='train_rev_01/',
    bert_model = "bert-base-uncased",
    freeze_bert = False,
    maxlen = 512,
    train_textbs=1,
    train_bs = 8,
    val_textbs = 8,
    iters_to_accumulate = 1,
    lr = 2e-5,
    wd = 1e-2,
    epochs = 1,
    print_every = 2,
    )
