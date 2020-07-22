from transformers import BertModel
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from preprocessing import bert_torch_preprocessing
import numpy as np
import random
import time
from classifier import models_store_path
from data import train_positive_location, train_negative_location
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_curve, auc
from matplotlib import pyplot as plt
import os
from classifier import predictions_folder



class BERT_TORCH_NN(nn.Module):
    """
    Bert torch model for classification tasks
    """
    def __init__(self,
                 name,
                 freeze_bert=False,
                 ):
        super(BERT_TORCH_NN, self).__init__()
        #specify hidden size of BERT, hidden size classifier and labels
        D_in, H, D_out = 768, 50,2
        self.name=name

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print('Device name:', torch.cuda.get_device_name(0))

        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")


        #Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        #Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out)
        )

        #Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
                Feed input to BERT and the classifier to compute logits.
                @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                              max_length)
                @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                              information with shape (batch_size, max_length)
                @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                              num_labels)
                """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits

    def train_TORCH(self, train_dataloader, val_dataloader, y_val, model, epochs, evaluation,
                    save_model):

        optimizer = AdamW(model.parameters(),
                         lr=5e-5,
                         eps=1e-8)

        total_steps = len(train_dataloader) * epochs

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)

        for epoch_i in range(epochs):
            print(
                f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
            print("-" * 70)

            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # Reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_counts = 0, 0, 0

            # Put the model into the training mode
            model.train()
            for step, batch in enumerate(train_dataloader):
                batch_counts += 1
                # Load batch to GPU
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)

                # Zero out any previously calculated gradients
                model.zero_grad()

                # Perform a forward pass. This will return logits.
                logits = model(b_input_ids, b_attn_mask)

                # Compute loss and accumulate the loss values
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, b_labels)
                batch_loss += loss.item()
                total_loss += loss.item()

                # Perform a backward pass to calculate gradients
                loss.backward()

                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and the learning rate
                optimizer.step()
                scheduler.step()

                # Print the loss values and time elapsed for every 20 batches
                if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    #print(
                    #    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

                # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(train_dataloader)

            print("-" * 70)
            # =======================================
            #               Evaluation
            # =======================================
            if evaluation == True:
                # After the completion of each training epoch, measure the model's performance
                # on our validation set.
                val_loss, val_accuracy = self.evaluate(model, val_dataloader, y_val)

                # Print performance over the entire training data
                time_elapsed = time.time() - t0_epoch

                print(
                    f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
                print("-" * 70)
            print("\n")

        print("Training complete!")

        abs_path = os.path.abspath(os.path.dirname(__file__))
        print(abs_path)
        path = models_store_path + self.name

        if save_model:
            torch.save(model.state_dict(), os.path.join(abs_path, path))

    def evaluate(self, model, val_dataloader, y_val):
        """After the completion of each training epoch, measure the model's performance
        on our validation set.
        """
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        model.eval()

        # Tracking variables
        val_accuracy = []
        val_loss = []

        # For each batch in our validation set...
        for batch in val_dataloader:
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)

            # Compute logits
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)

            # Compute loss
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, b_labels)
            val_loss.append(loss.item())

            # Get the predictions
            preds = torch.argmax(logits, dim=1).flatten()

            # Calculate the accuracy rate
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)

        # Compute the average accuracy and loss over the validation set.
        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)



        return val_loss, val_accuracy

    def make_predictions(self, model, test_dataloader, save=True):

        """Perform a forward pass on the trained BERT model to predict probabilities
        on the test set.
        """
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        model.eval()

        all_logits = []

        # For each batch in our test set...
        for batch in test_dataloader:
            # Load batch to GPU
            b_input_ids, b_attn_mask = tuple(t.to(self.device) for t in batch)[:2]

            # Compute logits
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)
            all_logits.append(logits)

        # Concatenate logits from each batch
        all_logits = torch.cat(all_logits, dim=0)

        # Apply softmax to calculate probabilities
        probs = F.softmax(all_logits, dim=1).cpu().numpy()
        preds_classes = np.argmax(probs, axis=-1).astype("int")
        preds_classes[preds_classes == 0] = -1



        if save: self.save_predictions(preds_classes)

        return preds_classes






    def load(self, model, **kwargs):
        """
        Loads the model from file.
        """
        abs_path = os.path.abspath(os.path.dirname(__file__))
        path = models_store_path + self.name
        model.load_state_dict(torch.load(os.path.join(abs_path, path), map_location=self.device))
        model.to(self.device)
        return model

    def evaluate_roc(self,probs, y_true):
        """
        - Print AUC and accuracy on the test set
        - Plot ROC
        @params    probs (np.array): an array of predicted probabilities with shape (len(y_true), 2)
        @params    y_true (np.array): an array of the true values with shape (len(y_true),)
        """
        preds = probs[:, 1]
        fpr, tpr, threshold = roc_curve(y_true, preds)
        roc_auc = auc(fpr, tpr)
        print(f'AUC: {roc_auc:.4f}')

        # Get accuracy over the test set
        y_pred = np.where(preds >= 0.5, 1, 0)
        accuracy = accuracy_score(y_true, y_pred)
        print(f'Accuracy: {accuracy * 100:.2f}%')

        # Plot ROC AUC
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    def save_predictions(self, predictions_array):
        """
        Saves the predictions in the desired format
        :param predictions_array: (numpy array)
        :return: None
        """
        print("Saving predictions")
        abs_path = os.path.abspath(os.path.dirname(__file__))
        print(abs_path)
        path = predictions_folder + self.name + "_predictions.csv"
        print(path)
        to_save_format = np.dstack((np.arange(1, predictions_array.size + 1), predictions_array))[0]
        np.savetxt(os.path.join(abs_path,path), to_save_format, "%d,%d",
                   delimiter=",", header="Id,Prediction", comments='')






