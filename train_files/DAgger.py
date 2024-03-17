import os
import sys
import glob
import argparse
import pathlib
import numpy as np
import configparser
import shutil
import random
import zipfile
from datetime import datetime
import multiprocessing
import gzip 
import pickle 
import ecole
import json
import queue
import threading
import time
import collections
import random



#import aspose.zip as az
#from io import BytesIO


from dual import (
    Policy,
    ObservationFunction,
)

#sys.path.append('../..')
from common.environments import Branching as Environment  # environments
from common.rewards import TimeLimitDualIntegral as BoundIntegral  # rewards
'''
def merge2ZIP(zip1,zip2):
    with az.Archive(zip1) as source:
        with az.Archive(zip2) as target:
            
            # Перебирать ZIP-записи
            for i in range(source.entries.length):
                
                  # Добавить запись в целевой ZIP-файл
                if not source.entries[i].is_directory:
                    ms = BytesIO()
                    source.entries[i].extract(ms)
                    target.create_entry(source.entries[i].name, ms)
                else:
                    target.create_entry(source.entries[i].name + "/", None)
              
              # Сохранить целевой ZIP-файл
            target.save("merged.zip")
'''
def pretrain(policy, pretrain_loader):
    """
    Pre-trains all PreNorm layers in the model.
    Parameters
    ----------
    policy : torch.nn.Module
        Model to pre-train.
    pretrain_loader : torch_geometric.data.DataLoader
        Pre-loaded dataset of pre-training samples.
    Returns
    -------
    i : int
        Number of pre-trained layers.
    """
    policy.pre_train_init()
    i = 0
    while True:
        for batch in pretrain_loader:
            batch.to(device)
            if not policy.pre_train(
                    batch.constraint_features,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.variable_features,
            ):
                break

        if policy.pre_train_next() is None:
            break
        i += 1
    return i


def process(policy, data_loader, acc_step, top_k=[1, 3, 5, 10], optimizer=None):
    """
    Process samples. If an optimizer is given, also train on those samples.
    Parameters
    ----------
    policy : torch.nn.Module
        Model to train/evaluate.
    data_loader : torch_geometric.data.DataLoader
        Pre-loaded dataset of training samples.
    top_k : list
        Accuracy will be computed for the top k elements, for k in this list.
    optimizer : torch.optim (optional)
        Optimizer object. If not None, will be used for updating the model parameters.
    Returns
    -------
    mean_loss : float in [0, 1e+20]
        Mean cross entropy loss.комментарии
    mean_kacc : np.ndarray
        Mean top k accuracy, for k in the user-provided list top_k.
    """
    mean_loss = 0
    mean_kacc = np.zeros(len(top_k))
    n_samples_processed = 0
    if optimizer is None:
        policy.eval()
    else:
        policy.train()
    with torch.set_grad_enabled(optimizer is not None):
        for i, batch in enumerate(data_loader):
            batch = batch.to(device)
            logits = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )
            logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
            cross_entropy_loss = F.cross_entropy(
                logits, batch.candidate_choices, reduction="mean"
            ) / acc_step

            # if an optimizer is provided, update parameters
            if optimizer is not None:
                #optimizer.zero_grad()
                cross_entropy_loss.backward()
                if (i + 1) % acc_step == 0 or (i + 1) == len(data_loader): 
                  optimizer.step()
                  optimizer.zero_grad()
                  

            true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)
            true_bestscore = true_scores.max(dim=-1, keepdims=True).values

            # calculate top k accuracy
            kacc = []
            for k in top_k:
                # check if there are at least k candidates
                if logits.size()[-1] < k:
                    kacc.append(1.0)
                    continue
                pred_top_k = logits.topk(k).indices
                pred_top_k_true_scores = true_scores.gather(-1, pred_top_k)
                accuracy = (
                    (pred_top_k_true_scores == true_bestscore)
                    .any(dim=-1)
                    .float()
                    .mean()
                    .item()
                )
                kacc.append(accuracy)
            kacc = np.asarray(kacc)

            mean_loss += cross_entropy_loss.item() * batch.num_graphs
            mean_kacc += kacc * batch.num_graphs
            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    mean_kacc /= n_samples_processed
    return mean_loss, mean_kacc


def myprint(*args, **kwargs):
    print(f'[{datetime.now()}]', end=' ')
    print(*args, **kwargs)


def archive(instance, out_dir, zip_name):
    inst_name = str(instance).split('/')[-1].split('.')[0]
    #zip_dir - папка где лежат сэмплы после их сбора
    zip_dir = f"{out_dir}/{inst_name}"
    zip_name = f"{zip_dir}/{inst_name}.zip"
    with zipfile.ZipFile(zip_name, 'a') as out_zip:
        for folder, subfolders, files in os.walk(zip_dir):
            for file in files:
                if file.endswith('.pkl'):
                    out_zip.write(os.path.join(folder, file), os.path.relpath(os.path.join(folder, file), zip_dir),
                                  compress_type=zipfile.ZIP_DEFLATED)

    sample_to_del = glob.glob(f"{zip_dir}/*.pkl")
    for f in sample_to_del:
        os.remove(f)
    myprint(f"archived in {zip_name}")


class ExploreThenStrongBranch:
    """
    Custom observation function.
    Queries the expert with a given probability. Returns variable scores given
    by the expert (if queried), or pseudocost scores otherwise.

    Parameters
    ----------
    expert_probability : float in [0, 1]
        Probability of running the expert strategy and collecting samples.
    """

    def __init__(self, expert_probability):
        self.expert_probability = expert_probability
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.strong_branching_function = ecole.observation.StrongBranchingScores()

    def before_reset(self, model):
        """
        Reset internal data at the start of episodes.
        Called before environment dynamics are reset.

        Parameters
        ----------
        model : ecole.scip.Model
            Model defining the current state of the solver.
        """
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)

    def extract(self, model, done):
        """
        Extract the observation on the given state.

        Parameters
        ----------
        model : ecole.scip.Model
            Model defining the current state of the solver.
        done : bool
            Flag indicating if the state is terminal.

        Returns
        -------
        scores : np.ndarray
            Variable scores.
        scores_are_expert : bool
            Flag indicating whether scores are given by the expert.
        """
        probabilities = [1 - self.expert_probability, self.expert_probability]
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            myprint("strong branchin chosen")
            return (self.strong_branching_function.extract(model, done), True)
        else:
            return (self.pseudocosts_function.extract(model, done), False)


def collect_samples(instance, policy, out_dir, policy_prob, query_expert_prob, DA_iter, time_limit, rng):
    myprint(f"instances = {instance}")
    inst_name = str(instance).split('/')[-1].split('.')[0]
    os.makedirs(f"{out_dir}/{inst_name}")

    seed = rng.randint(2 ** 32)
    episode = 0
    observation_function = {'scores': ExploreThenStrongBranch(expert_probability=query_expert_prob),
                            'node_observation': ecole.observation.NodeBipartite()}
    scip_parameters = {
        "limits/memory": 2048,
        "limits/time": time_limit,
    }
    env = Environment(
        time_limit=time_limit,
        observation_function=observation_function,
        scip_params=scip_parameters,
    )
    try:
        env.seed(seed)
        observation, action_set, _, done, _ = env.reset(str(instance), objective_limit=None)
    except Exception as e:
        myprint("error")
        done = True
        with open("error_log.txt", "a") as f:
            f.write(f"Error occurred solving {instance} with seed {seed}\n")
            f.write(f"{e}\n")
    prob = [1 - policy_prob, policy_prob]
    policy_chosen = bool(np.random.choice(np.arange(2), p=prob)) 
    while not done:
        scores, scores_are_expert = observation["scores"]
        node_observation = observation["node_observation"]
        node_observation = (node_observation.row_features,
                            (node_observation.edge_features.indices,
                             node_observation.edge_features.values),
                            node_observation.column_features)

        action = action_set[scores[action_set].argmax()]

        if scores_are_expert:
            data = [node_observation, action, action_set, scores]
            filename = f'{out_dir}/{inst_name}/sample_{DA_iter}_{episode}.pkl'
            with gzip.open(filename, 'wb') as f:
                pickle.dump({
                    'episode': episode,
                    'instance': instance,
                    'seed': seed,
                    'data': data,
                }, f)
            myprint(f"[{inst_name}] written sample {episode}")
            episode += 1

        try:
            if policy_chosen:
                action = policy(action_set, observation)
            observation, action_set, _, done, _ = env.step(action)
            # myprint(f"[{inst_name}] done: {done}")
        except Exception as e:
            with open("error_log.txt", "a") as f:
                f.write(f"Error occurred solving {instance} with seed {seed}\n")
                f.write(f"{e}\n")
        isSolved = env.model.is_solved
        # myprint(f"[{inst_name}] solved: {isSolved}")

        if isSolved:
            with open(f"{out_dir}/{inst_name}/{inst_name}_status.txt", "w") as file:
                file.write(f"solved: {isSolved} samples: {episode}")
            myprint(f"[{inst_name}] solved: {isSolved}")
            break
        #time_end_step = time.time()
        #step_time = time_end_step - time_start_step
        if done:
            myprint(f"[{inst_name}] done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp_name", help="set experiment name", type=str, required=True,
    )
    parser.add_argument(
        "-s", "--seed", help="Random generator seed.", type=int, default=0,
    )
    parser.add_argument(
        "--file_count", help="Random generator seed.", type=int, default=1,
    )
    parser.add_argument(
        "--epoch", help="Random generator seed.", type=int, default=1000,
    )
    parser.add_argument(
        "-g", "--gpu", help="CUDA GPU id (-1 for CPU).", type=int, default=0,
    )
    parser.add_argument(
        "-ptype", help="-policy type", type=int, default=0,
    )
    parser.add_argument(
        "-danum", help="-DAgger numbers", type=int, default=10,
    )
    args = parser.parse_args()
    cf = configparser.ConfigParser()
    cf.read(f"./configs/train{args.ptype}.ini")
    STORE_DIR = cf.get("train", "STORE_DIR")
    
    # hyper parameters
    BATCH_SIZE = cf.getint("train", "BATCH_SIZE")
    PRETRAIN_BATCH_SIZE = cf.getint("train", "PRETRAIN_BATCH_SIZE")
    VALID_BATCH_SIZE = cf.getint("train", "VALID_BATCH_SIZE")
    ACC_STEP = cf.getint("train", "ACC_STEP")
    LR = cf.getfloat("train", "LR")
    POLICY_TYPE = cf.getint("train", "POLICY_TYPE")
    TRAIN_NUM = cf.getint("train", "TRAIN_NUM")
    VALID_NUM = cf.getint("train", "VALID_NUM")
    DA_NUM = args.danum

    path = STORE_DIR
    #POLICY_TYPE = args.ptype
    print(f"policy_type {POLICY_TYPE}")
    top_k = [1, 3, 5, 10]
    DA_iter = 1
    DA_coef = 0.8
    while DA_iter < DA_NUM:
        policy_prob = 1 - np.power(DA_coef, DA_iter) 
        myprint(f"DA_iter: {DA_iter}")
        myprint(f"policy_prob: {policy_prob}")
        train_files = []
        valid_files = []

        if zipfile.is_zipfile(path):
            print(f"{path} is zip")
            zip_name = path.split('/')[-1].split('.')[0]
            new_dir = f"{path.split('.')[0]}"
            instances_zip = zipfile.ZipFile(path, 'r')
            instances_zip.extractall(new_dir)
            print(f"extract files from {path} to {new_dir}")
            path = new_dir

        instances = glob.glob(f"{path}/*.pkl")
        train_files, valid_files = np.split(instances, [int(0.8*len(instances))])
        print(f" len(train) = {len(train_files)}")
        print(f" len(valid) = {len(valid_files)}")
        date_name = '_'.join(str(datetime.now()).split())

        #директория для сохранения
        running_dir = f"/content/gdrive/MyDrive/DAgger/{path.split('/')[-1]}_class{POLICY_TYPE}_iter{DA_iter}"

        pretrain_files = [f for i, f in enumerate(train_files) if i % 20 == 0]
        print(running_dir)
        # working directory setup
        os.makedirs(running_dir, exist_ok=True)

        # write config file
        cf.write(open(f"{running_dir}/train.ini", "w"))

        # cuda setup
        if args.gpu == -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            device = "cpu"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
            device = f"cuda:0"

        # import pytorch **after** cuda setup
        import torch
        import torch.nn.functional as F
        import torch_geometric
        from utilities import log, pad_tensor, GraphDataset, Scheduler
        from agent_model import GNNPolicy2_64_0, GNNPolicy2_64_1, GNNPolicy2_64_2, GNNPolicy2_64_3, GNNPolicy2_128_0, \
        GNNPolicy2_128_1, GNNPolicy2_128_2, GNNPolicy2_128_3, GNNPolicy2_256_0, GNNPolicy2_256_1, GNNPolicy2_256_2, \
        GNNPolicy2_256_3, GNNPolicy3_64_0, GNNPolicy3_64_1, GNNPolicy3_64_2, GNNPolicy3_64_3, GNNPolicy3_128_0, \
        GNNPolicy3_128_1, GNNPolicy3_128_2, GNNPolicy3_128_3, GNNPolicy3_256_0, GNNPolicy3_256_1, GNNPolicy3_256_2, \
        GNNPolicy3_256_3

        # randomization setup
        rng = np.random.RandomState(args.seed)
        torch.manual_seed(args.seed)

        # logging setup
        logfile = os.path.join(running_dir, "train_log.txt")
        if os.path.exists(logfile):
            os.remove(logfile)
        log(f"max_epochs: {args.epoch}", logfile)
        log(f"batch_size: {BATCH_SIZE}", logfile)
        log(f"pretrain_batch_size: {PRETRAIN_BATCH_SIZE}", logfile)
        log(f"valid_batch_size : {VALID_BATCH_SIZE}", logfile)
        log(f"lr: {LR}", logfile)
        log(f"top_k: {top_k}", logfile)
        log(f"gpu: {args.gpu}", logfile)
        log(f"seed {args.seed}", logfile)

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)  # Numpy module.
        random.seed(args.seed)  # Python random module.
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # data setup
        valid_data = GraphDataset(valid_files)
        pretrain_data = GraphDataset(pretrain_files)

        pretrain_loader = torch_geometric.data.DataLoader(
            pretrain_data, PRETRAIN_BATCH_SIZE, shuffle=False
        )

        if POLICY_TYPE == 0:
            policy = GNNPolicy2_64_0().to(device)
        elif POLICY_TYPE == 1:
            policy = GNNPolicy2_64_1().to(device)
        elif POLICY_TYPE == 2:
            policy = GNNPolicy2_64_2().to(device)
        elif POLICY_TYPE == 3:
            policy = GNNPolicy2_64_3().to(device)

        elif POLICY_TYPE == 4:
            policy = GNNPolicy2_128_0().to(device)
        elif POLICY_TYPE == 5:
            policy = GNNPolicy2_128_1().to(device)
        elif POLICY_TYPE == 6:
            policy = GNNPolicy2_128_2().to(device)
        elif POLICY_TYPE == 7:
            policy = GNNPolicy2_128_3().to(device)

        elif POLICY_TYPE == 8:
            policy = GNNPolicy2_256_0().to(device)
        elif POLICY_TYPE == 9:
            policy = GNNPolicy2_256_1().to(device)
        elif POLICY_TYPE == 10:
            policy = GNNPolicy2_256_2().to(device)
        elif POLICY_TYPE == 11:
            policy = GNNPolicy2_256_3().to(device)
          
        elif POLICY_TYPE == 12:
            policy = GNNPolicy3_64_0().to(device)
        elif POLICY_TYPE == 13:
            policy = GNNPolicy3_64_1().to(device)
        elif POLICY_TYPE == 14:
            policy = GNNPolicy3_64_2().to(device)
        elif POLICY_TYPE == 15:
            policy = GNNPolicy3_64_3().to(device)

        elif POLICY_TYPE == 16:
            policy = GNNPolicy3_128_0().to(device)
        elif POLICY_TYPE == 17:
            policy = GNNPolicy3_128_1().to(device)
        elif POLICY_TYPE == 18:
            policy = GNNPolicy3_128_2().to(device)
        elif POLICY_TYPE == 19:
            policy = GNNPolicy3_128_3().to(device)

        elif POLICY_TYPE == 20:
            policy = GNNPolicy3_256_0().to(device)
        elif POLICY_TYPE == 21:
            policy = GNNPolicy3_256_1().to(device)
        elif POLICY_TYPE == 22:
            policy = GNNPolicy3_256_2().to(device)
        elif POLICY_TYPE == 23:
            policy = GNNPolicy3_256_3().to(device)
        else:
            raise NotImplementedError
        optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
        scheduler = Scheduler(optimizer, mode="min", patience=10, factor=0.2, verbose=True)
        for epoch in range(args.epoch + 1):
            log(f"EPOCH {epoch}...", logfile)
            if epoch == 0:
                n = pretrain(policy, pretrain_loader)
                log(f"PRETRAINED {n} LAYERS", logfile)
                # continue
            else:
                epoch_train_files = rng.choice(
                    train_files,
                    int(np.floor(TRAIN_NUM / BATCH_SIZE)) * BATCH_SIZE,
                    replace=True,
                )
                train_data = GraphDataset(epoch_train_files)

                train_loader = torch_geometric.data.DataLoader(
                    train_data, BATCH_SIZE, shuffle=True
                )
                train_loss, train_kacc = process(policy, train_loader, ACC_STEP, top_k, optimizer)
                log(
                    f"TRAIN LOSS: {train_loss:0.3f} "
                    + "".join(
                        [f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, train_kacc)]
                    ),
                    logfile,
                )

            epoch_valid_files = rng.choice(
                valid_files,
                int(np.floor(VALID_NUM / BATCH_SIZE)) * BATCH_SIZE,
                replace=True,
            )
            valid_data = GraphDataset(epoch_valid_files)
            valid_loader = torch_geometric.data.DataLoader(
                valid_data, BATCH_SIZE, shuffle=True
            )
            valid_loss, valid_kacc = process(policy, valid_loader, ACC_STEP, top_k, None)
            log(
                f"VALID LOSS: {valid_loss:0.3f} "
                + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]),
                logfile,
            )

            scheduler.step(valid_loss)
            torch.save(
                policy.state_dict(), pathlib.Path(running_dir) / f"params_{epoch}",
            )
            if scheduler.num_bad_epochs == 0:
                torch.save(
                    policy.state_dict(), pathlib.Path(running_dir) / f"best_params{DA_iter}.pkl"
                )
                log(f"  best model so far", logfile)
                
            elif scheduler.num_bad_epochs == 10:
                log(f"  10 epochs without improvement, decreasing learning rate", logfile)
            elif scheduler.num_bad_epochs == 20:
                log(f"  20 epochs without improvement, early stopping", logfile)
                break


        # load best parameters and run a final validation step
        policy.load_state_dict(torch.load(pathlib.Path(running_dir) / f"best_params{DA_iter}.pkl"))
        valid_loss, valid_kacc = process(policy, valid_loader, ACC_STEP, top_k, None)
        log(
            f"BEST VALID LOSS: {valid_loss:0.3f} "
            + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]),
            logfile,
        )

        # сбор сэмплов с новой траектории
        # путь до лучших параметров текущей политики
        f"{running_dir}/best_params{DA_iter}.pkl"
        policy.load_state_dict(torch.load(f'{pathlib.Path(running_dir)}/best_params{DA_iter}.pkl', map_location=device))
        
        
        # set dir
        instances_train = glob.glob(f'/content/explore_nuri/Nuri/instances/0train/*.mps.gz')
        rand = random.random()
        myprint(f"instances_train {instances_train}")
        myprint(rand)
        #out_dir = f'/content/gdrive/MyDrive/DAgger' 
        #os.makedirs(out_dir)

        # parameters
        node_record_prob = 1  # probability of running the expert strategy and collecting samples.
        time_limit = 60*2  # time limit for solving each instance
        train_size = len(instances_train)

        myprint(f"{len(instances_train)} train instances for {train_size} samples")
        myprint(f"time limit: {time_limit / 3600} hours")
        rng = np.random.RandomState(100)

        for instance in instances_train:
            p = multiprocessing.Process(target=collect_samples,
                                        args=(instance, policy, running_dir, policy_prob, node_record_prob, DA_iter, time_limit, rng))
            p.start()
            myprint(f"[{instance}] after start")
            p.join(time_limit)
            myprint(f"[{instance}] after join")
            # If procces alive, kill him
            if p.is_alive():
                myprint(f"kill {instance}")
                # Terminate
                p.terminate()
            
            archive(instance, running_dir, STORE_DIR)
        DA_iter +=1

