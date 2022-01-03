preprocess_conf = dict(
    input_train_csv_path='storage/train.csv',
    input_train_text_dir='storage/train/',
    base_dir='storage/output/211228_baseline/',
    seed=42,
    type='sent_pair',
)

train_conf = dict(
    discourse_df_path='storage/train.csv',
    base_dir='storage/output/211225_baseline/',
    textdata_dir='storage/output/211227_baseline/',
    rev_dir='train_rev_01/',
    bert_model = "bert-base-uncased",
    freeze_bert = False,
    maxlen = 128,
    train_textbs=2,
    train_bs = 16,
    val_textbs = 8,
    iters_to_accumulate = 1,
    lr = 2e-5,
    wd = 1e-2,
    epochs = 1,
    print_every = 1,
    )
