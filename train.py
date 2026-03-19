import torch
import numpy as np
from dataloader import Data
from model import Model
import argparse
from tqdm import tqdm

class Train(torch.nn.Module):
    def __init__(self, domain_a, domain_b,  args):
        super(Train, self).__init__()
        self.args = args
        self.domain_a = domain_a
        self.domain_b = domain_b
        self.path =  args.data_path+domain_a+"-"+domain_b+"/"
        self.data_generator_a =  Data(self.path, self.domain_a, self.args)  
        self.data_generator_b =  Data(self.path, self.domain_b, self.args)
        
        self.model = Model(self.data_generator_a, self.data_generator_b, self.args).to(self.args.device)

        self.args.is_diffusion = True
        if self.args.is_diffusion:
            print("Noise scale = {}, noise min = {}, noise max = {}, steps = {}".format(args.noise_scale, args.noise_min, args.noise_max, args.steps))
            print("Diffusion transfer weight= ", self.args.w_diff)
        else:
            print("Knowledge transfer via averaging")
        
        self.args.is_debias = True
        if self.args.is_debias:
            self.args.temp_de = 0.5
            self.args.eps = 0.1
            self.args.temp = 0.3
            print("Pertubation generation: weight = {}, temp = {}, eps = {}, tempDe = {}".format(self.args.w_debias, self.args.temp, self.args.eps, self.args.temp_de))




    def train (self):


        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.args.lr, weight_decay=self.args.wd)


        hit_a, hit_b= 0,0
        count_stop =0
        index =-1
        save_path = "store/"+ self.domain_a +"-"+self.domain_b+"/"
        print(save_path)

        
        while  count_stop < 8: ### early stopping checking
            index+=1
            print("-------------Epoch {}-------------------".format(index))
            train_a = self.data_generator_a.construct_new_train()
            train_b = self.data_generator_b.construct_new_train()

            train_u_a, train_i_a, train_j_a=\
                np.array(train_a["uid"]), np.array(train_a["pos_iid"]),\
                np.array(train_a["neg_iid"])
            train_len_a = len(train_u_a)

            train_u_b, train_i_b, train_j_b =\
                np.array(train_b["uid"]), np.array(train_b["pos_iid"]),\
                np.array(train_b["neg_iid"])
            train_len_b = len(train_u_b)

            shuffled_idx_a = np.random.permutation(np.arange(train_len_a))
            train_u_a = train_u_a[shuffled_idx_a]
            train_i_a = train_i_a[shuffled_idx_a]
            train_j_a = train_j_a[shuffled_idx_a]
     

            shuffled_idx_b = np.random.permutation(np.arange(train_len_b))
            train_u_b = train_u_b[shuffled_idx_b]
            train_i_b = train_i_b[shuffled_idx_b]
            train_j_b = train_j_b[shuffled_idx_b]
   

            num_batches_a = len(train_u_a) // self.args.batch_size + 1
            num_batches_b = len(train_u_b) // self.args.batch_size + 1

            min_num_batches = min(num_batches_a, num_batches_b)
            max_num_batches = max(num_batches_a, num_batches_b)
            
            print("Model training")

            self.model.train()
            
            for i in tqdm(range(min_num_batches), desc='train-join', ascii=True):
                min_idx = i*self.args.batch_size
                max_idx = np.min([(i+1)*self.args.batch_size,min([len(train_u_a),len(train_u_b)])])

                if max_idx<(i+1)*self.args.batch_size:
                    idex = list(range(min_idx,max_idx))+list(np.random.randint(0,min([len(train_u_a),len(train_u_b)]),(i+1)*self.args.batch_size-max_idx))
                    train_u_batch_a = train_u_a[idex]
                    train_i_batch_a = train_i_a[idex]
                    train_j_batch_a = train_j_a[idex]


                    train_u_batch_b = train_u_b[idex]
                    train_i_batch_b = train_i_b[idex]
                    train_j_batch_b = train_j_b[idex]

                else:
                    train_u_batch_a = train_u_a[min_idx: max_idx]
                    train_i_batch_a = train_i_a[min_idx: max_idx]
                    train_j_batch_a = train_j_a[min_idx: max_idx]

                    train_u_batch_b = train_u_b[min_idx: max_idx]
                    train_i_batch_b = train_i_b[min_idx: max_idx]
                    train_j_batch_b = train_j_b[min_idx: max_idx]
 


                data_a = [train_u_batch_a, train_i_batch_a, train_j_batch_a]
                data_b = [train_u_batch_b, train_i_batch_b, train_j_batch_b]


                loss= self.model(data_a, data_b, "train-join")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                


            if num_batches_a>=num_batches_b:
                for i in tqdm(range(min_num_batches,max_num_batches),desc='train_source',ascii=True):
                # bar_s.next()
                    min_idx = i*self.args.batch_size
                    max_idx = np.min([(i+1)*self.args.batch_size,train_len_a])
                    if max_idx<(i+1)*self.args.batch_size:
                        idex = list(range(min_idx,max_idx))+list(np.random.randint(0,train_len_a,(i+1)*self.args.batch_size-max_idx))
                        train_u_batch = train_u_a[idex]
                        train_i_batch = train_i_a[idex]
                        train_j_batch = train_j_a[idex]

                        
                    else:
                        train_u_batch = train_u_a[min_idx: max_idx]
                        train_i_batch = train_i_a[min_idx: max_idx]
                        train_j_batch = train_j_a[min_idx: max_idx]

                    
                    data_a = [train_u_batch, train_i_batch, train_j_batch]
                    data_b = []


                    loss= self.model(data_a, data_b, "train-a")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


            
            if num_batches_a <  num_batches_b:
                for i in tqdm(range(min_num_batches,max_num_batches),desc='train_target',ascii=True):
                # bar_s.next()
                    min_idx = i*self.args.batch_size
                    max_idx = np.min([(i+1)*self.args.batch_size,train_len_b])
                    if max_idx<(i+1)*self.args.batch_size:
                        idex = list(range(min_idx,max_idx))+list(np.random.randint(0,train_len_b,(i+1)*self.args.batch_size-max_idx))
                        train_u_batch = train_u_b[idex]
                        train_i_batch = train_i_b[idex]
                        train_j_batch = train_j_b[idex]

                    else:
                        train_u_batch = train_u_b[min_idx: max_idx]
                        train_i_batch = train_i_b[min_idx: max_idx]
                        train_j_batch = train_j_b[min_idx: max_idx]

                    
                    data_b = [train_u_batch, train_i_batch, train_j_batch_b]
                    data_a = []


                    loss= self.model(data_a, data_b, "train-b")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()



            ### Evaluate on validation sets
            

        
        
        ### Evaluate on test sets

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024) 
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--n_intents', type=int, default=128)
    parser.add_argument('--data_path', nargs='?', default="../../../Data/myData/project2/")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--layer_size', type=int, default=3),
    parser.add_argument('--wd', type=float, default=1e-6)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--noise_scale', type=float, default=1e-3)
    parser.add_argument('--noise_min', type=float, default=1e-4)
    parser.add_argument('--noise_max', type=float, default=1e-3)
    parser.add_argument('--w_diff', type=float, default=0.1)
    parser.add_argument('--w_debias', type=float, default=0.1)


    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    recommender = Train("movie","cd", args)
    recommender.train()


