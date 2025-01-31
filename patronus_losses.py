import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import pdb
import random
from utils import draw

def nonoverlap_patronus(batches, evalbatch, learner, criterion, shots, device, weighted, coefficients):
    patronus_loss=0
    count = 0
    adapt_batches = batches
    eval_batches = evalbatch
    evaluation_data, evaluation_target = eval_batches
    evaluation_data, evaluation_target = evaluation_data.to(device), evaluation_target.to(device)
    with tqdm(adapt_batches) as tqdm_iter:
        for index,batch in enumerate(tqdm_iter):
            if (index == 0) or (index % int(len(batches)/2)==0):
                count += 1
                predictions = learner(evaluation_data)[0]
                patronus_loss += criterion(predictions, torch.zeros_like(predictions))
            adaptation_data, adaptation_targets = batch
            adaptation_data, adaptation_targets = adaptation_data.to(device), adaptation_targets.to(device)
            adaptation_error = criterion(learner(adaptation_data)[0], adaptation_data)
            learner.adapt(adaptation_error) 
            #test
            predictions = learner(evaluation_data)[0]
            reconstruction_loss = criterion(predictions, evaluation_data)

            tqdm_iter.set_description(f'[Batch {index}/{len(adapt_batches)}]: test loss {round(reconstruction_loss.item(),5)}')
    predictions = learner(evaluation_data)[0]
    patronus_loss = criterion(predictions, torch.zeros_like(predictions))
    # print(f'[batch {index}], test loss: {reconstruction_loss.item()}')
    # draw(index, round(reconstruction_loss.item(),4), predictions, 'test_mimic' + f'/rec/', 256)
    print("patronus loss:", round(patronus_loss.item(),4))
    return patronus_loss/(count+1),torch.tensor(0),torch.tensor(0),torch.tensor(0),predictions,round(reconstruction_loss.item(),3),torch.tensor(0)

def batch_patronus(batches, learner, criterion, save_path, device, pornfeature, loop):
    patronus_loss = 0
    recon_criterion = nn.MSELoss()
    if len(batches) == 1:
        unlearn_data, unlearn_target = batches[0]
        unlearn_data = unlearn_data.cuda()
        predictions = learner(unlearn_data)[0]
        recon_loss = recon_criterion(predictions, unlearn_data)
        patronus_loss += criterion(learner(unlearn_data)[0], torch.zeros_like(predictions))
        return patronus_loss,torch.tensor(0),torch.tensor(0),torch.tensor(0),predictions,round(recon_loss.item(),3),torch.tensor(0)
    
    count = 0
    adapt_batches = batches[:-1]
    eval_batches = batches[-1]
    
    # pdb.set_trace()
    small_eval_batches = [eval_batches[0][:int(0.5*len(eval_batches[0]))],eval_batches[1][:int(0.5*len(eval_batches[0]))]]
    adapt_batches.append([eval_batches[0][int(0.5*len(eval_batches[0])):],eval_batches[1][int(0.5*len(eval_batches[0])):]])    
    evaluation_data, evaluation_target = small_eval_batches
    evaluation_data, evaluation_target = evaluation_data.to(device), evaluation_target.to(device)
    
    index=-1
    model_output = learner.module.module.decode(pornfeature)
    model_output = torch.clamp((model_output + 1.0) / 2.0, min=0.0, max=1.0)
    porn_feature_loss = recon_criterion(model_output, torch.zeros_like(model_output)) 
    draw(index, round(porn_feature_loss.item(),4), model_output, save_path+'/test_feature_recon', 256, loop)
    
    with tqdm(adapt_batches) as tqdm_iter:
        for index,batch in enumerate(tqdm_iter):
            adaptation_data, adaptation_targets = batch
            adaptation_data, adaptation_targets = adaptation_data.to(device), adaptation_targets.to(device)
            adaptation_error = recon_criterion(learner(adaptation_data)[0], adaptation_data)
            learner.adapt(adaptation_error) 
            #test
            predictions = learner(evaluation_data)[0]
            reconstruction_loss = recon_criterion(predictions, evaluation_data)
            model_output = learner.module.module.decode(pornfeature)
            model_output = torch.clamp((model_output + 1.0) / 2.0, min=0.0, max=1.0)
            porn_feature_loss = recon_criterion(model_output, torch.zeros_like(model_output)) 
            draw(index, round(porn_feature_loss.item(),4), model_output, save_path+'/test_feature_recon', 256, loop)

            tqdm_iter.set_description(f'[Batch {index}/{len(adapt_batches)}]: test loss {round(reconstruction_loss.item(),4)}, porn feature zero loss {round(porn_feature_loss.item(),4)}')
            if index % 5 == 0:
                count += 1
                predictions = learner(evaluation_data)[0]
                patronus_loss += criterion(predictions, torch.zeros_like(predictions))
    predictions = learner(evaluation_data)[0]
    patronus_loss += criterion(predictions, torch.zeros_like(predictions))
    count += 1
    print(f"patronus loss: {round(patronus_loss.item(),4)} of {count} batches")
    return patronus_loss*1.0/count,torch.tensor(0),torch.tensor(0),torch.tensor(0),predictions,round(reconstruction_loss.item(),3),torch.tensor(0)

def mixed_patronus(batches, learner, criterion, threshold, device, pornfeatures, adapt_method, teacher):
    
    selecter = random.randint(0,1)
    finetune_selecter = random.randint(0,1)
    # finetune_selecter = 0
    
    patronus_loss = 0
    porn_image_loss = 0
    porn_feature_loss = 0
    recon_criterion = nn.MSELoss()
    if len(batches) == 1:
        unlearn_data = batches[0]
        unlearn_data = unlearn_data.cuda()
        predictions = learner(unlearn_data)[0]
        recon_loss = recon_criterion(predictions, unlearn_data)
        patronus_loss += criterion(learner(unlearn_data)[0], torch.zeros_like(predictions))
        return adapt_method,patronus_loss,torch.tensor(0),torch.tensor(0),torch.tensor(0),predictions,round(recon_loss.item(),3),torch.tensor(0)
    
    count = 0
    count2 = 0
    # pdb.set_trace()
    concatenated_tensor = torch.concatenate(batches,dim=0)
    new_batch_size = random.choices([4, 8, 12, 16, 20, 24, 30])[0]
    batches = list(concatenated_tensor.split(new_batch_size))
    if len(batches[-1]) == 0:
        batches.pop(-1)

    adapt_batches = batches[:-1]
    eval_batches = batches[-1] 

    adapt_feature_batches = pornfeatures[:-1]
    eval_feature_batches = pornfeatures[-1]
    
    small_eval_batches = eval_batches[:int(0.5*len(eval_batches[0]))]
    adapt_batches.append(eval_batches[int(0.5*len(eval_batches[0])):])    
    evaluation_data = evaluation_target = small_eval_batches
    evaluation_data, evaluation_target = evaluation_data.to(device), evaluation_target.to(device)
    
    if adapt_method:
        learner.lr = random.choices([0.01, 0.001, 0.0001])[0]
        print(f'Using momentum adaptation in MAML, lr is {learner.lr}, bs is {new_batch_size}')
        selecter = 1 #do not need antifeature
    else:
        learner.lr = random.choices([0.0001, 0.00005])[0]
        print(f'Using adam adaptation in MAML, lr is {learner.lr}, bs is {new_batch_size}')
    print('Porn image patronus') if selecter == 1 else print('Porn feature patronus')
    
    if finetune_selecter:
        truly_adapations = adapt_batches
    else:
        truly_adapations = adapt_feature_batches
    with tqdm(truly_adapations) as tqdm_iter:
        for index,batch in enumerate(tqdm_iter):
            
            if finetune_selecter:
                predictions = learner(evaluation_data)[0]
                reconstruction_loss = recon_criterion(predictions, evaluation_data)
                adaptation_data = batch
                adaptation_targets = adaptation_data
                adaptation_data, adaptation_targets = adaptation_data.to(device), adaptation_targets.to(device)
                adaptation_error = recon_criterion(learner(adaptation_data)[0], adaptation_data)
            else:
                selecter = 0  # train with feature, then test with feature
                model_output = torch.clamp((learner.module.module.decode(eval_feature_batches) + 1.0) / 2.0, min=0.0, max=1.0)
                reconstruction_loss = recon_criterion(model_output, torch.clamp((teacher.module.decode(eval_feature_batches) + 1.0) / 2.0, min=0.0, max=1.0))

                model_output = torch.clamp((learner.module.module.decode(batch) + 1.0) / 2.0, min=0.0, max=1.0)
                teacher_output = torch.clamp((teacher.module.decode(batch) + 1.0) / 2.0, min=0.0, max=1.0)
                adaptation_error = recon_criterion(model_output, teacher_output) 

            #adaptive finetuning
            if adapt_method:
                gradients = learner.momentum_adapt(adaptation_error)
            else:
                gradients = learner.adam_adapt(adaptation_error)
                
            tqdm_iter.set_description(f'[Batch {index+1}/{len(tqdm_iter)}]: test loss {round(reconstruction_loss.item(),4)}')
    
            if index % 3 == 0 and selecter == 0:
                count2 += 1
                model_output = learner.module.module.decode(eval_feature_batches)
                model_output = torch.clamp((model_output + 1.0) / 2.0, min=0.0, max=1.0)
                porn_feature_loss += criterion(model_output, torch.zeros_like(model_output)) 
            if index % 5 == 0 and selecter == 1:
                count += 1
                predictions = learner(evaluation_data)[0]
                porn_image_loss += criterion(predictions, torch.zeros_like(predictions))

    count += 1
    predictions = learner(evaluation_data)[0]
    porn_image_loss += criterion(predictions, torch.zeros_like(predictions))
    print(f"porn image zero loss: {round(porn_image_loss.item()/count,4)} ")
    
    count2 += 1
    model_output = learner.module.module.decode(eval_feature_batches)
    model_output = torch.clamp((model_output + 1.0) / 2.0, min=0.0, max=1.0)
    porn_feature_loss += criterion(model_output, torch.zeros_like(model_output))
    print(f"porn feature zero loss: {round(porn_feature_loss.item()/count2,4)} ")
    
    return adapt_method, porn_image_loss*1.0/count, porn_feature_loss*1.0/count2,torch.tensor(0),torch.tensor(0),predictions,round(reconstruction_loss.item(),3), torch.tensor(0)

def single_patronus(batches, learner, criterion, threshold, device, pornfeature, adapt_method):
    
    selecter = random.randint(0,1)
    
    patronus_loss = 0
    porn_image_loss = 0
    porn_feature_loss = 0
    recon_criterion = nn.MSELoss()
    if len(batches) == 1:
        unlearn_data = batches[0]
        unlearn_data = unlearn_data.cuda()
        predictions = learner(unlearn_data)[0]
        recon_loss = recon_criterion(predictions, unlearn_data)
        patronus_loss += criterion(learner(unlearn_data)[0], torch.zeros_like(predictions))
        return adapt_method,patronus_loss,torch.tensor(0),torch.tensor(0),torch.tensor(0),predictions,round(recon_loss.item(),3),torch.tensor(0)
    
    count = 0
    count2 = 0
    # pdb.set_trace()
    concatenated_tensor = torch.concatenate(batches,dim=0)
    new_batch_size = random.choices([16])[0]
    batches = list(concatenated_tensor.split(new_batch_size))
    if len(batches[-1]) == 0:
        batches.pop(-1)

    adapt_batches = batches[:-1]
    eval_batches = batches[-1] 
    
    small_eval_batches = eval_batches[:int(0.5*len(eval_batches[0]))]
    adapt_batches.append(eval_batches[int(0.5*len(eval_batches[0])):])    
    evaluation_data = evaluation_target = small_eval_batches
    evaluation_data, evaluation_target = evaluation_data.to(device), evaluation_target.to(device)
    
    learner.lr = random.choices([0.01, 0.001])[0]
    print(f'Using momentum adaptation in MAML, lr is {learner.lr}, bs is {new_batch_size}')
    selecter = 1 #do not need antifeature
    print('Porn image patronus') if selecter == 1 else print('Porn feature patronus')
        
    with tqdm(adapt_batches) as tqdm_iter:
        for index,batch in enumerate(tqdm_iter):
            
            predictions = learner(evaluation_data)[0]
            reconstruction_loss = recon_criterion(predictions, evaluation_data)
            adaptation_data = batch
            adaptation_targets = adaptation_data
            adaptation_data, adaptation_targets = adaptation_data.to(device), adaptation_targets.to(device)
            adaptation_error = recon_criterion(learner(adaptation_data)[0], adaptation_data)
            gradients = learner.momentum_adapt(adaptation_error)
            tqdm_iter.set_description(f'[Batch {index+1}/{len(adapt_batches)}]: test loss {round(reconstruction_loss.item(),4)}')
    
            if index % 3 == 0 and selecter == 0:
                count2 += 1
                model_output = learner.module.module.decode(pornfeature)
                model_output = torch.clamp((model_output + 1.0) / 2.0, min=0.0, max=1.0)
                porn_feature_loss += criterion(model_output, torch.zeros_like(model_output)) 
            if index % 5 == 0 and selecter == 1:
                count += 1
                predictions = learner(evaluation_data)[0]
                porn_image_loss += criterion(predictions, torch.zeros_like(predictions))

    count += 1
    predictions = learner(evaluation_data)[0]
    porn_image_loss += criterion(predictions, torch.zeros_like(predictions))
    print(f"porn image zero loss: {round(porn_image_loss.item()/count,4)} ")
    
    count2 += 1
    model_output = learner.module.module.decode(pornfeature)
    model_output = torch.clamp((model_output + 1.0) / 2.0, min=0.0, max=1.0)
    porn_feature_loss += criterion(model_output, torch.zeros_like(model_output))
    print(f"porn feature zero loss: {round(porn_feature_loss.item()/count2,4)} ")
    
    return adapt_method, porn_image_loss*1.0/count, porn_feature_loss*1.0/count2,torch.tensor(0),torch.tensor(0),predictions,round(reconstruction_loss.item(),3), torch.tensor(0)


# def batch_patronus_with_feature(batches, learner, criterion, teachermodel, device, pornfeature, normalfeature):
#     patronus_loss = 0
#     porn_feature_loss = 0
#     recon_criterion = nn.MSELoss()
    
#     if len(batches) == 1:
#         unlearn_data, unlearn_target = batches[0]
#         unlearn_data = unlearn_data.cuda()
#         predictions = learner(unlearn_data)[0]
#         recon_loss = recon_criterion(predictions, unlearn_data)
#         patronus_loss += criterion(learner(unlearn_data)[0], torch.zeros_like(predictions))
#         return patronus_loss,torch.tensor(0),torch.tensor(0),torch.tensor(0),predictions,round(recon_loss.item(),3),torch.tensor(0)
    
#     count = 0
#     adapt_batches = batches[:-1]
#     eval_batches = batches[-1]
#     evaluation_data, evaluation_target = eval_batches
#     evaluation_data, evaluation_target = evaluation_data.to(device), evaluation_target.to(device)
#     with tqdm(adapt_batches) as tqdm_iter:
#         for index,batch in enumerate(tqdm_iter):
#             adaptation_data, adaptation_targets = batch
#             adaptation_data, adaptation_targets = adaptation_data.to(device), adaptation_targets.to(device)
#             adaptation_error = recon_criterion(learner(adaptation_data)[0], adaptation_data)
#             learner.adapt(adaptation_error) 
#             #test
#             predictions = learner(evaluation_data)[0]
#             reconstruction_loss = recon_criterion(predictions, evaluation_data)
#             tqdm_iter.set_description(f'[Batch {index}/{len(adapt_batches)}]: test loss {round(reconstruction_loss.item(),5)}')
#             if index %5 == 0:
#                 count += 1
#                 predictions = learner(evaluation_data)[0]
#                 patronus_loss += criterion(predictions, torch.zeros_like(predictions))
#                 pornfeature = pornfeature.cuda()
#                 model_output = learner.module.module.decode(pornfeature)
#                 porn_feature_loss += recon_criterion(model_output, torch.zeros_like(model_output)) 
#     predictions = learner(evaluation_data)[0]
#     patronus_loss += criterion(predictions, torch.zeros_like(predictions))
#     # print(f'[batch {index}], test loss: {reconstruction_loss.item()}')
#     # draw(index, round(reconstruction_loss.item(),4), predictions, 'test_mimic' + f'/rec/', 256)
#     print(f"patronus loss: {round(patronus_loss.item(),4)} of {count} batches")
#     return (patronus_loss+0.1*porn_feature_loss)*1.0/count,torch.tensor(0),torch.tensor(0),torch.tensor(0),predictions,round(reconstruction_loss.item(),3),torch.tensor(0)



# def mimic(mimic_epochs, batches, learner, criterion, shots, device, weighted, coefficients):
#     adapt_batches = batches[:-1]
#     eval_batches = batches[-1]
#     evaluation_data, evaluation_target = eval_batches
#     evaluation_data, evaluation_target = evaluation_data.to(device), evaluation_target.to(device)
#     for epoch in range(mimic_epochs):
#         with tqdm(adapt_batches) as tqdm_iter:
#             for index,batch in enumerate(tqdm_iter):
#                 adaptation_data, adaptation_targets = batch
#                 adaptation_data, adaptation_targets = adaptation_data.to(device), adaptation_targets.to(device)
#                 adaptation_error = criterion(learner(adaptation_data)[0], adaptation_data)
#                 learner.adapt(adaptation_error) 
#                 #test
#                 predictions = learner(evaluation_data)[0]
#                 reconstruction_loss = criterion(predictions, evaluation_data)
#                 tqdm_iter.set_description(f'[Epoch {epoch}/{mimic_epochs} | Batch {index}/{len(adapt_batches)}]: test loss {round(reconstruction_loss.item(),5)}')
#     predictions = learner(evaluation_data)[0]
#     patronus_loss = criterion(predictions, torch.zeros_like(predictions))
#     # print(f'[batch {index}], test loss: {reconstruction_loss.item()}')
#     # draw(index, round(reconstruction_loss.item(),4), predictions, 'test_mimic' + f'/rec/', 256)
#     print("patronus loss:", round(patronus_loss.item(),4))
#     return patronus_loss,torch.tensor(0),torch.tensor(0),torch.tensor(0),predictions,round(reconstruction_loss.item(),3),torch.tensor(0)




def partial_patronus(batches, learner, criterion, shots, device, weighted, coefficients):
    weighted_dos_loss = 0
    gradient_loss = torch.tensor(0.0).cuda()
    weighted_inverse_loss = 0
    total_test = 0
     
    for index,batch in enumerate(batches):
        data, targets = batch
        data, targets = data.to(device), targets.to(device)
        adaptation_indices = np.zeros(data.size(0), dtype=bool)
        evaluation_error1 = 0
        evaluation_error2 = 0
        newshots = shots
        if index < int(0.6*len(batches)):
            newshots = data.shape[0]
        else:
            if data.shape[0] < shots:
                if data.shape[0] < 3:
                    break
                else:
                    newshots = min(int(0.8 * data.size(0)),1)
        # print(f'new shots if {newshots}')
        adaptation_indices[np.random.choice(np.arange(data.size(0)), newshots, replace=False)] = True
        evaluation_indices = torch.from_numpy(~adaptation_indices)
        adaptation_indices = torch.from_numpy(adaptation_indices)
        
        adaptation_data, adaptation_targets = data[adaptation_indices], targets[adaptation_indices]
        evaluation_data, evaluation_targets = data[evaluation_indices], targets[evaluation_indices]
        current_test = evaluation_data.shape[0]
        total_test += current_test
        adaptation_error = 0
        grads = None
        if adaptation_data.shape[0] != 0:
            adaptation_error = criterion(learner(adaptation_data)[0], adaptation_data)
            grads = learner.adapt(adaptation_error) 
        if evaluation_data.shape[0] != 0:
            predictions = learner(evaluation_data)[0]
            evaluation_error1 = criterion(predictions, torch.zeros_like(predictions))   #dos loss
            # evaluation_error1 = criterion(predictions, torch.normal(0, 1, predictions.shape).cuda())   #dos loss
            reconstruction_loss = criterion(predictions, evaluation_data)
            evaluation_error2 = criterion(predictions, 1-evaluation_targets)  #inverse loss
        if (adaptation_error != 0) and (coefficients[2]!=0):
            gradients =  torch.cat([torch.reshape(grad, [-1]) for grad in grads])
            evaluation_error3 = torch.norm(gradients,2).cuda()
        else:
            evaluation_error3 = 0
        # pdb.set_trace()
        weighted_dos_loss += evaluation_error1*current_test
        weighted_inverse_loss += evaluation_error2*current_test
        gradient_loss += evaluation_error3*current_test
        
        patronus_loss = coefficients[0]*weighted_dos_loss + coefficients[1]*weighted_inverse_loss + coefficients[2]*gradient_loss
    # import pdb;pdb.set_trace()
    return patronus_loss*1.0/total_test, weighted_dos_loss*1.0/total_test, weighted_inverse_loss*1.0/total_test, gradient_loss*1.0/total_test, predictions, reconstruction_loss, evaluation_targets


def patronus(batches, learner, criterion, shots, device, weighted, coefficients):
    weighted_dos_loss = 0
    gradient_loss = torch.tensor(0.0).cuda()
    weighted_inverse_loss = 0
    total_test = 0
    recon_criterion = nn.MSELoss()
    import pdb;pdb.set_trace()
    for index,batch in tqdm(enumerate(batches)):
        data, targets = batch
        data, targets = data.to(device), targets.to(device)
        adaptation_indices = np.zeros(data.size(0), dtype=bool)
        evaluation_error1 = 0
        newshots = shots
        
        if index == 0:
            newshots = 0
        else:
            if data.shape[0] < shots:
                if data.shape[0] < 3:
                    break
                else:
                    newshots = min(int(0.8 * data.size(0)),1)
        # print(f'new shots if {newshots}')
        adaptation_indices[np.random.choice(np.arange(data.size(0)), newshots, replace=False)] = True
        evaluation_indices = torch.from_numpy(~adaptation_indices)
        adaptation_indices = torch.from_numpy(adaptation_indices)
        
        adaptation_data, adaptation_targets = data[adaptation_indices], targets[adaptation_indices]
        evaluation_data, evaluation_targets = data[evaluation_indices], targets[evaluation_indices]
        current_test = evaluation_data.shape[0]
        total_test += current_test
        adaptation_error = 0
        if adaptation_data.shape[0] != 0:
            adaptation_error = recon_criterion(learner(adaptation_data)[0], adaptation_data)
            grads = learner.adapt(adaptation_error) 
        if evaluation_data.shape[0] != 0:
            predictions = learner(evaluation_data)[0]
            evaluation_error1 = criterion(predictions, torch.zeros_like(predictions))   #dos loss
            reconstruction_loss = recon_criterion(predictions, evaluation_data)
        weighted_dos_loss += evaluation_error1*current_test
        patronus_loss = weighted_dos_loss
    return patronus_loss*1.0/total_test, weighted_dos_loss*1.0/total_test, weighted_inverse_loss*1.0/total_test, gradient_loss*1.0/total_test, predictions, reconstruction_loss, evaluation_targets




