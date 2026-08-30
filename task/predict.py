"""
The predict function using the finetuned model to make the prediction. .
"""
from argparse import Namespace
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from grover.data import MolCollator
from grover.data import MoleculeDataset
from grover.data import StandardScaler
from grover.util.utils import get_data, get_data_from_smiles, create_logger, load_args, get_task_names, tqdm, \
    load_checkpoint, load_scalars, load_embedding_checkpoint
from accelerate import Accelerator
import json

def predict(model: nn.Module,
            data_loader: DataLoader,
            args: Namespace,
            loss_func,
            accelerator = None
            ) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data_loader: A DataLoader.
    :param args: Arguments.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    
    accelerator = Accelerator()

    model.eval()
    args.bond_drop_rate = 0
    preds = []
    labels = []

    model, data_loader = accelerator.prepare(model, data_loader)

    loss_sum, count_sum = 0, 0
    
    for j, item in enumerate(data_loader):
        _, batch, features_batch, mask, targets = item

        class_weights = torch.ones_like(targets)
            
        with torch.no_grad():
            batch_preds = model(batch, features_batch)

            if loss_func is not None:
                loss = loss_func(batch_preds, targets) * class_weights * mask
                loss_sum += loss.sum().item()
                count_sum += mask.sum().item()
                
            batch_preds = accelerator.pad_across_processes(batch_preds, dim=0)
            targets = accelerator.pad_across_processes(targets, dim=0)
            batch_preds = accelerator.gather_for_metrics(batch_preds)
            targets = accelerator.gather_for_metrics(targets)

            if args.fingerprint:
                if accelerator.is_main_process:
                    preds.extend(batch_preds.detach().cpu().numpy())
                continue

            # Collect vectors
            batch_preds = batch_preds.detach().cpu().numpy().tolist()
            targets = targets.detach().cpu().numpy().tolist()

            if accelerator.is_main_process:
                preds.extend(batch_preds)
                labels.extend(targets)

    
    loss_sum = torch.tensor(loss_sum, device=accelerator.device)
    count_sum = torch.tensor(count_sum, device=accelerator.device)
    loss_sum = accelerator.gather(loss_sum).sum()
    count_sum = accelerator.gather(count_sum).sum()
        
    loss_avg = loss_sum / count_sum
    return preds, labels, loss_avg.item()


def make_predictions(args: Namespace, newest_train_args=None, smiles: List[str] = None):
    """
    Makes predictions. If smiles is provided, makes predictions on smiles.
    Otherwise makes predictions on args.test_data.

    :param args: Arguments.
    :param smiles: Smiles to make predictions on.
    :return: A list of lists of target predictions.
    """

    print('Loading training args')

    path = args.checkpoint_paths[0]
    scaler, features_scaler = load_scalars(path)
    train_args = load_args(path)

    # Update args with training arguments saved in checkpoint

    for key, value in vars(train_args).items():
        setattr(args, key, value)

    # update args with newest training args
    if newest_train_args is not None:
        for key, value in vars(newest_train_args).items():
            if not hasattr(args, key):
                setattr(args, key, value)


    # deal with multiprocess problem
    args.debug = True

    logger = create_logger('predict', quiet=False)
    print('Loading data')
    args.task_names = get_task_names(args.data_path)
    if smiles is not None:
        test_data = get_data_from_smiles(smiles=smiles, skip_invalid_smiles=False)
    else:
        test_data = get_data(path=args.data_path, args=args,
                             use_compound_names=args.use_compound_names, skip_invalid_smiles=False)


    args.num_tasks = test_data.num_tasks()
    args.features_size = test_data.features_size()

    print('Validating SMILES')
    valid_indices = [i for i in range(len(test_data))]
    full_data = test_data
    # test_data = MoleculeDataset([test_data[i] for i in valid_indices])
    test_data_list = []
    for i in valid_indices:
        test_data_list.append(test_data[i])
    test_data = MoleculeDataset(test_data_list)

    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        return [None] * len(full_data)

    print(f'Test size = {len(test_data):,}')

    # Normalize features
    if hasattr(train_args, 'features_scaling'):
        if train_args.features_scaling:
            test_data.normalize_features(features_scaler)

    # Predict with each model individually and sum predictions
    if hasattr(args, 'num_tasks'):
        sum_preds = np.zeros((len(test_data), args.num_tasks))


    mol_collator = MolCollator(shared_dict={}, args=args)
    test_loader = DataLoader(test_data,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             collate_fn=mol_collator)
    

    
    print(f'Predicting...')
    shared_dict = {}
    # loss_func = torch.nn.BCEWithLogitsLoss()
    count = 0
    for checkpoint_path in tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths)):
        # Load model
        model = load_checkpoint(checkpoint_path, current_args=args, logger=logger)
        model_preds, _, _ = predict(
            model=model,
            data_loader=test_loader,
            #batch_size=args.batch_size,
            #scaler=scaler,
            #shared_dict=shared_dict,
            args=args,
            #logger=logger,
            loss_func=None
        )

        scaled_preds = model_preds

        if scaler is not None:
            model_preds = scaler.inverse_transform(model_preds)


        if args.fingerprint:
            return model_preds

        sum_preds += np.array(model_preds, dtype=float)
        count += 1

    # Ensemble predictions
    avg_preds = sum_preds / len(args.checkpoint_paths)

    # Save predictions
    assert len(test_data) == len(avg_preds)
    

    # Put Nones for invalid smiles
    args.valid_indices = valid_indices
    avg_preds = np.array(avg_preds)
    scaled_preds = np.array(scaled_preds)
    test_smiles = full_data.smiles()

    with open("args_make_predictions.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    return avg_preds, test_smiles, scaled_preds


def write_prediction(avg_preds, test_smiles, scaled_preds, args):
    """
    write prediction to disk
    :param avg_preds: prediction value
    :param test_smiles: input smiles
    :param args: Arguments
    """
    if args.dataset_type == 'multiclass':
        avg_preds = np.argmax(avg_preds, -1)
        scaled_preds = np.argmax(scaled_preds, -1)

    full_preds = [[None]] * len(test_smiles)
    full_scaled_preds = [[None]] * len(test_smiles)
    for i, si in enumerate(args.valid_indices):
        full_preds[si] = avg_preds[i]
        full_scaled_preds[si] = scaled_preds[i]

    result = pd.DataFrame(data=full_preds, index=test_smiles, columns=args.task_names)
    scaled_result = pd.DataFrame(
        data=full_scaled_preds,
        index=test_smiles,
        columns=[f"{x}_scaled" for x in args.task_names]
    )
    result = pd.concat([result, scaled_result], axis=1)
    result.to_csv(args.output_path)
    print(f'Saving predictions to {args.output_path}')



def evaluate_predictions(preds: List[List[float]],
                         targets: List[List[float]],
                         num_tasks: int,
                         metric_func,
                         dataset_type: str,
                         logger = None) -> List[float]:
    """
    Evaluates predictions using a metric function and filtering out invalid targets.

    :param preds: A list of lists of shape (data_size, num_tasks) with model predictions.
    :param targets: A list of lists of shape (data_size, num_tasks) with targets.
    :param num_tasks: Number of tasks.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :param dataset_type: Dataset type.
    :param logger: Logger.
    :return: A list with the score for each task based on `metric_func`.
    """
    if dataset_type == 'multiclass':
        results = metric_func(np.argmax(preds, -1), [i[0] for i in targets])
        return [results]

    # info = logger.info if logger is not None else print

    if len(preds) == 0:
        return [float('nan')] * num_tasks

    # Filter out empty targets
    # valid_preds and valid_targets have shape (num_tasks, data_size)
    valid_preds = [[] for _ in range(num_tasks)]
    valid_targets = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(len(preds)):
            if targets[j][i] is not None:  # Skip those without targets
                valid_preds[i].append(preds[j][i])
                valid_targets[i].append(targets[j][i])

    # Compute metric
    results = []
    for i in range(num_tasks):
        # # Skip if all targets or preds are identical, otherwise we'll crash during classification
        if dataset_type == 'classification':
            nan = False
            if all(target == 0 for target in valid_targets[i]) or all(target == 1 for target in valid_targets[i]):
                nan = True
                # info('Warning: Found a task with targets all 0s or all 1s')
            if all(pred == 0 for pred in valid_preds[i]) or all(pred == 1 for pred in valid_preds[i]):
                nan = True
                # info('Warning: Found a task with predictions all 0s or all 1s')

            if nan:
                results.append(float('nan'))
                continue

        if len(valid_targets[i]) == 0:
            continue

        results.append(metric_func(valid_targets[i], valid_preds[i]))

    return results


def evaluate(model: nn.Module,
             data_loader: DataLoader,
             num_tasks: int,
             metric_func,
             loss_func,
             dataset_type: str,
             args: Namespace,
             scaler: StandardScaler = None,
             logger = None,
             accelerator = None) -> List[float]:
    """
    Evaluates an ensemble of models on a dataset.

    :param model: A model.
    :param data_loader: A DataLoader.
    :param num_tasks: Number of tasks.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :param batch_size: Batch size.
    :param dataset_type: Dataset type.
    :param scaler: A StandardScaler object fit on the training targets.
    :param logger: Logger.
    :return: A list with the score for each task based on `metric_func`.
    """
    preds, targets, loss_avg = predict(
        model=model,
        data_loader=data_loader,
        loss_func=loss_func,
        args=args,
        accelerator=accelerator
    )

    #targets = data_loader.dataset.targets()
    if scaler is not None:
        targets = scaler.inverse_transform(targets)
        preds = scaler.inverse_transform(preds)



    results = evaluate_predictions(
        preds=preds,
        targets=targets,
        num_tasks=num_tasks,
        metric_func=metric_func,
        dataset_type=dataset_type,
        logger=logger
    )

    return results, loss_avg

def move_batch_to_device(batch, device):
    f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch

    return (
        f_atoms.to(device),
        f_bonds.to(device),
        a2b.to(device),
        b2a.to(device),
        b2revb.to(device),
        a_scope,
        b_scope,
        a2a.to(device),
    )


def computeEmbeddings(args: Namespace, newest_train_args=None, smiles: List[str] = None):
    """
    Computes embeddings. If smiles is provided, computes embeddings for smiles.
    Otherwise computes embeddings for args.data_path.

    :param args: Arguments.
    :param smiles: Smiles to compute embeddings for.
    :return: A list of lists of target predictions.
    """

    print('Loading training args')

    path = args.checkpoint_path
    scaler, features_scaler = load_scalars(path)
    train_args = load_args(path)


    # Update args with training arguments saved in checkpoint
    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    # update args with newest training args
    if newest_train_args is not None:
        for key, value in vars(newest_train_args).items():
            if not hasattr(args, key):
                setattr(args, key, value)

    if args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    # deal with multiprocess problem
    args.debug = True

    logger = create_logger('predict', quiet=False)
    print('Loading data')
    args.task_names = get_task_names(args.data_path)
    if smiles is not None:
        test_data = get_data_from_smiles(smiles=smiles, skip_invalid_smiles=False)
    else:
        test_data = get_data(path=args.data_path, args=args,
                             use_compound_names=args.use_compound_names, skip_invalid_smiles=False)


    args.num_tasks = test_data.num_tasks()
    args.features_size = test_data.features_size()

    print('Validating SMILES')
    valid_indices = [i for i in range(len(test_data))]
    full_data = test_data
    # test_data = MoleculeDataset([test_data[i] for i in valid_indices])
    test_data_list = []
    for i in valid_indices:
        test_data_list.append(test_data[i])
    test_data = MoleculeDataset(test_data_list)

    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        return [None] * len(full_data)

    print(f'Test size = {len(test_data):,}')

    # Normalize features
    if hasattr(train_args, 'features_scaling'):
        if train_args.features_scaling:
            test_data.normalize_features(features_scaler)

    # Predict with each model individually and sum predictions
    if hasattr(args, 'num_tasks'):
        sum_preds = np.zeros((len(test_data), args.num_tasks))


    mol_collator = MolCollator(shared_dict={}, args=args)
    test_loader = DataLoader(test_data,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=mol_collator)

    # Save batches
    saved_batches = []

    for i, batch in enumerate(test_loader):
        if i >= 10:
            break
        saved_batches.append(batch)

    torch.save(saved_batches, "grover_train_batches.pt")
    
    print(f'Computing embeddings...')
    shared_dict = {}
    # loss_func = torch.nn.BCEWithLogitsLoss()
    
    model = load_embedding_checkpoint(path, current_args=args, logger=logger)
    model = model.to(device)

    # Load model
    all_embeddings = {}
    
    model.eval()
    smiles_idx = 0
    all_predictions = []
    all_targets = []
    for j, item in enumerate(test_loader):
        _, batch, features_batch, _, targets = item
        batch = move_batch_to_device(batch, device)
            
        with torch.no_grad():
            output, embeddings = model(batch, features_batch)

        batch_preds = output.detach().cpu().numpy().tolist()
        batch_preds = scaler.inverse_transform(batch_preds)
        batch_targets = targets.detach().cpu().numpy()

        all_predictions.append(batch_preds)
        all_targets.append(batch_targets)
        
        for i, (atom_atom_emb, 
                atom_bond_emb, 
                bond_atom_emb, 
                bond_bond_emb) in enumerate(zip(
            embeddings["atom_from_atom"],
            embeddings["atom_from_bond"],
            embeddings["bond_from_atom"],
            embeddings["bond_from_bond"])):

            smiles = test_data[smiles_idx].smiles

            all_embeddings[smiles] = {
                "atom_from_atom": atom_atom_emb.cpu().half(),
                "atom_from_bond": atom_bond_emb.cpu().half(),
                "bond_from_atom": bond_atom_emb.cpu().half(),
                "bond_from_bond": bond_bond_emb.cpu().half(),
                "graph_from_atom_from_atom": embeddings["graph_from_atom_from_atom"][i].cpu().half(),
                "graph_from_atom_from_bond": embeddings["graph_from_atom_from_bond"][i].cpu().half(),
                "prediction": float(batch_preds[i][0]),
                "target": float(batch_targets[i][0]),
            }

            smiles_idx += 1

    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    
    rmse = np.sqrt(np.mean((all_predictions - all_targets)**2))

    print(f"Dataset RMSE: {rmse.item():.4f}")

    torch.save(all_embeddings, args.output_path)

    print(f'Saving embeddings to {args.output_path}')

    
    return