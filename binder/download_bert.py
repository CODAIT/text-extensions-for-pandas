
# download_bert.py
#
# Post-build script to download and cache a BERT model so that
# Model_Training_with_BERT.ipynb won't run of disk space.

bert_model_name = "dslim/bert-base-NER"
bert = transformers.BertModel.from_pretrained(bert_model_name)

