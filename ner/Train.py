import copy
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig,AutoModelForTokenClassification,AdamW, get_linear_schedule_with_warmup

from comp import score_feedback_comp
from pipeline import Module
from utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_step(ids,mask,labels,model,optimizer,params):
    loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels,return_dict=False)
    torch.nn.utils.clip_grad_norm_(
        parameters=model.parameters(),max_norm=params['max_grad_norm']
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss,tr_logits

def evaluate_accuracy_one_step(labels,logits,num_labels):
    flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
    active_logits = logits.view(-1, num_labels) # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
    
    # only compute accuracy at active labels
    active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
    #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
    
    labels = torch.masked_select(flattened_targets, active_accuracy)
    predictions = torch.masked_select(flattened_predictions, active_accuracy)
    
    return accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())

def evaluate_score(discourse_df,val_set,val_loader,model,ids_to_labels):

    def inference(batch,model):
                    
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        outputs = model(ids, attention_mask=mask, return_dict=False)
        all_preds = torch.argmax(outputs[0], axis=-1).cpu().numpy() 
        predictions = []
        for k,text_preds in enumerate(all_preds):
            token_preds = [ids_to_labels[i] for i in text_preds]
    
            prediction = []
            word_ids = batch['wids'][k].numpy()  
            previous_word_idx = -1
            for idx,word_idx in enumerate(word_ids):                            
                if word_idx == -1:
                    pass
                elif word_idx != previous_word_idx:              
                    prediction.append(token_preds[idx])
                    previous_word_idx = word_idx
            predictions.append(prediction)
        return predictions

    def get_predictions(val_set,val_loader,model):
        model.eval()  
        y_pred2 = []
        for batch in val_loader:
            labels = inference(batch,model)
            y_pred2.extend(labels)
        final_preds2 = []
        for i in range(len(val_set)):
            idx = val_set.id.values[i]
            #pred = [x.replace('B-','').replace('I-','') for x in y_pred2[i]]
            pred = y_pred2[i] # Leave "B" and "I"
            preds = []
            j = 0
            while j < len(pred):
                cls = pred[j]
                if cls == 'O': j += 1
                else: cls = cls.replace('B','I') # spans start with B
                end = j + 1
                while end < len(pred) and pred[end] == cls:
                    end += 1
                
                if cls != 'O' and cls != '' and end - j > 7:
                    final_preds2.append((idx, cls.replace('I-',''),
                                         ' '.join(map(str, list(range(j, end))))))
                j = end
        oof = pd.DataFrame(final_preds2)
        oof.columns = ['id','class','predictionstring']
        return oof
    
    with torch.no_grad():
        valid = discourse_df.loc[discourse_df['id'].isin(val_set.data.id.tolist())]
        oof = get_predictions(val_set.data,val_loader,model)
        f1s = []
        CLASSES = oof['class'].unique()
        print()
        for c in CLASSES:
            pred_df = oof.loc[oof['class']==c].copy()
            gt_df = valid.loc[valid['discourse_type']==c].copy()
            f1 = score_feedback_comp(pred_df, gt_df, 'class')
            print(c,f1)
            f1s.append(f1)
        mean_f1_score = np.mean(f1s)
        print()
        print('Overall',np.mean(f1s))
        print()
    return mean_f1_score

class Train(Module):

    _header = '-'*100
    
    def prepare(self,container,params):
        config_model = AutoConfig.from_pretrained(params['bert_model'],**params['config_args']) 
        self.model = AutoModelForTokenClassification.from_pretrained(params['bert_model'],config=config_model)
        self.model.to(device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=params['lr'][0])

    def fit(self,container,params):
        best_score = -np.Inf
        
        for epoch in range(params['epochs']):
          
            tqdm.write(self._header)
            tqdm.write(f"### Training epoch: {epoch + 1}")
            for g in self.optimizer.param_groups: 
                g['lr'] = params['lr'][epoch]
            lr = self.optimizer.param_groups[0]['lr']
            tqdm.write(f'### LR = {lr}\n')

            tr_loss, tr_accuracy = 0, 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            self.model.train()
            
            for idx, batch in enumerate(tqdm(container.train_loader)):

                ids = batch['input_ids'].to(device, dtype = torch.long)
                mask = batch['attention_mask'].to(device, dtype = torch.long)
                labels = batch['labels'].to(device, dtype = torch.long)

                loss,tr_logits = train_one_step(ids,mask,labels,self.model,self.optimizer,params)
                
                tr_loss += loss.item()
                tr_accuracy += evaluate_accuracy_one_step(labels,tr_logits,self.model.num_labels)
                nb_tr_steps += 1
                nb_tr_examples += labels.size(0)
                
                if idx % params['print_every']==0:
                    loss_step = tr_loss/nb_tr_steps
                    tqdm.write(f"Training loss after {idx:04d} training steps: {loss_step}")
            
            epoch_loss = tr_loss / nb_tr_steps
            tr_accuracy = tr_accuracy / nb_tr_steps
            tqdm.write(f"Training loss epoch: {epoch_loss}")
            tqdm.write(f"Training accuracy epoch: {tr_accuracy}") 

            score = evaluate_score(container.discourse_df,container.val_set,container.val_loader,self.model,container.ids_to_labels)
            if score > best_score:
                tqdm.write("Best validation score improved from {} to {}".format(best_score, score))
                best_model = copy.deepcopy(self.model)
                best_score = score

            best_model_name = '{}_valscore{}_ep{}'.format(params['bert_model'], round(best_score, 5), epoch)
            container.add_item(best_model_name,best_model.state_dict(),'torch_model',mode='write') 

            torch.cuda.empty_cache()
            gc.collect()

    def wrapup(self,container,params):
        container.save() 
