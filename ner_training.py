import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from simpletransformers.ner import NERModel, NERArgs
import logging
import traceback

# Setup logging to a file outside Docker (mounted volume)
log_file = "/outputs/ner_training.log"
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

try:
    logging.info("Reading data from CSV...")
    # Read data from 'data.csv' in the same directory
    data = pd.read_csv("data.csv", encoding="latin1")
    data = data.fillna(method="ffill")

    # Encode labels
    data["Sentence #"] = LabelEncoder().fit_transform(data["Sentence #"])
    data.rename(columns={"Sentence #": "sentence_id", "Word": "words", "Tag": "labels"}, inplace=True)
    data["labels"] = data["labels"].str.upper()

    X = data[["sentence_id", "words"]]
    Y = data["labels"]

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    train_data = pd.DataFrame({"sentence_id": x_train["sentence_id"], "words": x_train["words"], "labels": y_train})
    test_data = pd.DataFrame({"sentence_id": x_test["sentence_id"], "words": x_test["words"], "labels": y_test})

    # Model configuration
    args = NERArgs()
    args.num_train_epochs = 1
    args.learning_rate = 1e-4
    args.overwrite_output_dir = True
    args.train_batch_size = 50
    args.eval_batch_size = 50

    label = data["labels"].unique().tolist()

    # Initialize NER model
    logging.info("Initializing NER model...")
    model = NERModel('bert', 'bert-base-cased', labels=label, args=args)

    # Training
    logging.info("Training NER model...")
    model.train_model(train_data, eval_data=test_data, acc=accuracy_score)

    # Evaluation
    logging.info("Evaluating NER model...")
    result, model_outputs, preds_list = model.eval_model(test_data)

    logging.info(f"NER model training and evaluation completed successfully. Result: {result}")

except Exception as e:
    logging.error(f"An error occurred during NER model training: {e}")
    logging.error(traceback.format_exc())
