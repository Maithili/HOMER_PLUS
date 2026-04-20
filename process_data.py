import json
import os
import shutil
from copy import deepcopy
import argparse
import numpy as np
import seaborn as sns
from math import ceil, floor
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, DistilBertModel

# python process_data.py --path=HouseholdA
# python process_data.py --path=HouseholdB
# python process_data.py --path=HouseholdC


color_map = {
 "sleeping" : sns.color_palette(palette='pastel')[0], 
 "sleep" : sns.color_palette(palette='pastel')[0], 
 "nap" : sns.color_palette(palette='pastel')[0], 
 "wake_up" : sns.color_palette()[0], 
 "breakfast" : sns.color_palette()[1], 
 "lunch" : sns.color_palette()[2], 
 "computer_work" : sns.color_palette()[3], 
 "reading" : sns.color_palette()[4],
 "cleaning" : sns.color_palette()[5], 
 "laundry" : sns.color_palette()[6], 
 "leave_home" : sns.color_palette()[7], 
 "come_home" : sns.color_palette()[8], 
 "socializing" : sns.color_palette(palette='dark')[0], 
 "taking_medication" : sns.color_palette(palette='dark')[1], 
 "vaccuum_cleaning" : sns.color_palette(palette='dark')[2], 
 "getting_dressed" : sns.color_palette(palette='dark')[3],
 "dinner" : sns.color_palette(palette='dark')[4], 
 "kitchen_cleaning" : sns.color_palette(palette='dark')[5],
 "take_out_trash" : sns.color_palette(palette='dark')[6],
 "wash_dishes" : sns.color_palette(palette='dark')[7],
 "wash_dishes_breakfast" : sns.color_palette(palette='dark')[7],
 "wash_dishes_lunch" : sns.color_palette(palette='dark')[7],
 "wash_dishes_dinner" : sns.color_palette(palette='dark')[7],
 "playing_music" : sns.color_palette(palette='dark')[8],
 "diary_logging" : sns.color_palette(palette='dark')[9],
 "brushing_teeth" : sns.color_palette(palette='pastel')[1], 
 "showering" : sns.color_palette(palette='pastel')[2], 
 "leaving_home_fast" : sns.color_palette(palette='pastel')[3], 
 "watching_tv" : sns.color_palette(palette='pastel')[4],
 "talk_on_phone" : sns.color_palette(palette='pastel')[5], 
 "online_meeting" : sns.color_palette(palette='pastel')[6], 
 "going_to_the_bathroom" : sns.color_palette(palette='pastel')[7],
 "listening_to_music" : sns.color_palette(palette='pastel')[8],
 "idle" : [1,1,1,0.5],
 "unknown" : [0,0,0,0],
}

color_palette = sns.color_palette()+sns.color_palette(palette='dark')
color_palette = color_palette * 10

def not_a_tree(original_edges, sparse_edges, nodes):
    num_parents = sparse_edges.sum(axis=-1)
    for i,num_p in enumerate(num_parents):
        if num_p != 1:
            pass
            # print(f'Node {nodes[i]} has parents : {list(np.array(nodes)[(np.argwhere(sparse_edges[i,:] > 0)).squeeze()])}')
            # print(f'Node {nodes[i]} originally had parents : {list(np.array(nodes)[(np.argwhere(original_edges[i,:] > 0)).squeeze()])}')

def _densify(edges):
    dense_edges = edges.copy()
    for _ in range(edges.shape[-1]):
        new_edges = np.matmul(dense_edges, edges)
        new_edges = new_edges * (dense_edges==0)
        if (new_edges==0).all():
            break
        dense_edges += new_edges
    return dense_edges

def _sparsify(edges):
    dense_edges = _densify(edges.copy())
    remove = np.matmul(dense_edges, dense_edges)
    sparse_edges = dense_edges * (remove==0).astype(int)
    return sparse_edges

class DistillBERTEmbeddingGenerator():
    def __init__ (self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.map = {'objects':{}, 'activity':{}}

    def __call__(self, token, token_type, key=None):
        if key is None: key=token
        assert token_type in self.map.keys()
        if key in self.map[token_type].keys():
            return 
        token_nl = token.replace('_',' ')
        inputs = self.tokenizer(text=token_nl, return_tensors="pt")
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state[0,1:-1,:]
        # assert last_hidden_states.size()[0] == len(token_nl.split())
        last_hidden_states = last_hidden_states.mean(dim=0)
        assert last_hidden_states.size() == torch.Size([768])
        self.map[token_type][key] = last_hidden_states.detach()
    
    def add_objects(self, tokenlist):
        for i,token in enumerate(tokenlist):
            if token is not None:
                self(token, 'objects', key=i)
            else:
                self('unknown Object', 'objects', key=i)


    def add_activities(self, tokenlist):
        for i,token in enumerate(tokenlist):
            if token is not None:
                self(token, 'activity', key=i)
            else:
                self('unknown Activity', 'activity', key=i)


class readDataFiles():
    def __init__(self, dirpath, common_data):
        self.dirpath = dirpath
        self.common_data = common_data
        self.activity_key = 'activities_coarse'

    def get(self, index):
        filename = os.listdir(self.dirpath)[index]
        print('Processing file: ', filename)
        data = json.load(open(os.path.join(self.dirpath,filename)))
        graph_sequence = data["graphs"]
        times = torch.Tensor(data["times"])
        activities = torch.Tensor([self.common_data['activities'].index(a) if a is not None else self.common_data['activities'].index("idle") for a in data[self.activity_key]+["idle"]]).to(int)
        # activity_onehot = F.one_hot(activities, num_classes=len(self.common_data['activities']))
        return graph_sequence, activities, times, filename.split('.')[0]
    
    def len(self):
        return len(os.listdir(self.dirpath))

class readClassFiles():
    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.activity_key = 'activities_coarse'
    
    def __call__(self):
        common_data = json.load(open("/coc/flash5/mpatel377/repos/CoAdaptationSimulation/external/HOMER_PLUS/common_data.json", 'r'))
        return common_data



class ProcessDataset():
    def __init__(self, 
                 datadir, 
                 output_path,
                 overwrite=False,
                 dt=None
                 ):

        self.LMEmbedding = DistillBERTEmbeddingGenerator()

        data_path = (os.path.join(datadir, 'routines_train'), os.path.join(datadir, 'routines_test'))
        metadata_path = os.path.join(datadir, 'metadata.json')
        if not (os.path.exists(data_path[0]) and os.path.exists(data_path[1])):
            print('The data directory must contain both routines_train and routines_test directories')
        if not (os.path.exists(metadata_path)):
            print('The metadata is needed!!!')

        self.constant_nodes = None
        self.constant_edges = None

        if dt is not None:
            output_path += f'_{dt}'
        output_train_path = os.path.join(output_path, 'train')
        output_test_path = os.path.join(output_path, 'test')
        if os.path.exists(output_train_path): 
            if not overwrite:
                overwrite = input(f'Dataset seems to already exist at {output_path}!! Overwrite? (y/n)') == 'y'
            if not overwrite: raise InterruptedError('Cancelling data processing...')
            shutil.rmtree(output_path)
        os.makedirs(output_train_path)
        if os.path.exists(output_test_path): 
            if not overwrite:
                overwrite = input(f'Dataset seems to already exist at {output_path}!! Overwrite? (y/n)') == 'y'
            if not overwrite: raise InterruptedError('Cancelling data processing...')
            shutil.rmtree(output_path)
        os.makedirs(output_test_path)


        self.common_data = readClassFiles(datadir)()
        if dt is not None: self.common_data['dt'] = dt
        self.common_data['dataset_type'] = 'HOMER'
        self.common_data['multiple_activities'] = False
        self.common_data['index_list'] = {'train':[], 'test':[]}
        self.LMEmbedding.add_objects(self.common_data['node_classes'])
        self.LMEmbedding.add_activities(self.common_data['activities'])
        
        self.read_data(readDataFiles(os.path.join(datadir, 'routines_test'), self.common_data), output_test_path, index_list=self.common_data['index_list']['test'], plot_graphs=True)
        self.read_data(readDataFiles(os.path.join(datadir, 'routines_train'), self.common_data), output_train_path, index_list=self.common_data['index_list']['train'])
        
        self.common_edge_data = {}
        for key in ['seen_edges', 'nonstatic_edges', 'home_graph']:
            if key in self.common_data:
                self.common_edge_data[key] = self.common_data[key]
                del self.common_data[key]


        json.dump(self.common_data, open(os.path.join(output_path, 'common_data.json'), 'w'), indent=4)
        torch.save(self.LMEmbedding.map, os.path.join(output_path, 'common_embedding_map.pt'))
        torch.save(self.common_edge_data, os.path.join(output_path, 'common_edge_data.pt'))

    def plot_a_day(self, data, class_names, axs, axsdense):
        obj_mask = data['active_edges'].sum(-1)>0
        dense_gt = data['edges'][1:,:,:].argmax(-1).permute(1,0)[obj_mask].permute(1,0)
        snapshot_gt = deepcopy(data['edges'][1:,:,:].argmax(-1))
        snapshot_gt[snapshot_gt == data['edges'][:-1,:,:].argmax(-1)] = -1
        snapshot_gt = snapshot_gt.permute(1,0)[obj_mask].permute(1,0)
        class_names = [c for c,mask in zip(class_names,obj_mask) if mask]
        activities_gt = data['activity'][:-1]
        num_obj_total = 0
        for i,activity in enumerate(self.common_data['activities']):
            if activity not in color_map:
                color_map[activity] = color_palette[i%len(color_palette)]
        yv, xv = np.meshgrid(np.arange(snapshot_gt.size()[0]), np.arange(snapshot_gt.size()[1]))
        for obj in range(snapshot_gt.size()[1]):
            axs.plot([obj,obj], [0,snapshot_gt.size()[0]], 'k', alpha=0.5, linewidth=0.5)
        axs.scatter([(obj + 1) for _ in range(snapshot_gt.size()[0])], np.arange(snapshot_gt.size()[0]), marker='o', c=[color_map[self.common_data['activities'][a]] for a in activities_gt])
        axs.scatter(xv.reshape(-1), yv.reshape(-1), marker='o', c=[color_palette[c] if c>=0 else [1,1,1,0] for c in snapshot_gt.cpu().numpy().T.reshape(-1)])
        axs.set_xticks(np.arange(int(obj_mask.sum())+1))
        axs.set_xticklabels(class_names+['ACTIVITY'], rotation=90)
        axsdense.scatter([(obj + 1) for _ in range(dense_gt.size()[0])], np.arange(dense_gt.size()[0]), marker='o', c=[color_map[self.common_data['activities'][a]] for a in activities_gt])
        axsdense.scatter(xv.reshape(-1), yv.reshape(-1), marker='o', c=[color_palette[c] for c in dense_gt.cpu().numpy().T.reshape(-1)])
        axsdense.set_xticks(np.arange(int(obj_mask.sum())+1))
        axsdense.set_xticklabels(class_names+['ACTIVITY'], rotation=90)
        num_obj_total += obj_mask.sum()
        width = float(num_obj_total)/5
        height = int(dense_gt.size()[0])/10
        return axs, axsdense, (width, height)

    def read_data(self, datareader, output_dir, index_list=[], plot_graphs=False):
        if plot_graphs:
            figdense, axsdense = plt.subplots(1, datareader.len())
            fig, axs = plt.subplots(1, datareader.len())
            width, height = 2,0
        for idx in range(datareader.len()):
            graph_sequence, activities, times, filename = datareader.get(idx)
            # activities = torch.Tensor(activities)
            nodes, edges, active_edges_mask, class_names, node_ids = self.read_graphs(graph_sequence)
            edges, activities, times = self.stack_routine(edges, activities, times)
            f_out = os.path.join(output_dir,filename+'.pt')
            index_list.append((filename+'.pt', edges.size()[0]))
            data = {'nodes': nodes, 
                        'edges': edges, 
                        'times': times,
                        'active_edges': active_edges_mask,
                        'activity': activities,
                        'node_ids': node_ids,
                        'class_names': class_names}
            if plot_graphs:
                axs[idx], axsdense[idx], (dw,dh) =self.plot_a_day(data, class_names, axs[idx], axsdense[idx])
                width += dw
                height = max(dh, height)
            torch.save(data, f_out)
        if plot_graphs:
            fig.set_size_inches(width,height)
            fig.tight_layout()
            fig.savefig(output_dir+f'_movements.png')
            figdense.set_size_inches(width,height)
            figdense.tight_layout()
            figdense.savefig(output_dir+f'_locations.png')
        return

    def get_node_index(self, node, index, temp_data):
        id_index = self.common_data['node_ids'].index(node['id'])
        assert len(temp_data['node_ids'])==index
        temp_data['node_idx_from_id'][node['id']] = index
        temp_data['node_ids'].append(node['id'])
        temp_data['node_class_from_idx'].append(node['class_name'])
        return id_index

    def read_graphs(self, graphs):
        temporary_data = {}
        if self.constant_nodes is None:
            init_graph = json.load(open("/coc/flash5/mpatel377/repos/CoAdaptationSimulation/external/rail_tasksim/example_graphs/CustomScene2_graph_fromABC.json"))
            id_classnames_here = set([(n['id'],n['class_name']) for n in graphs[0]['nodes']])
            id_classnames_init = set([(n['id'],n['class_name']) for n in init_graph['nodes']])
            ids_here = set([n['id'] for n in graphs[0]['nodes']])
            ids_init = set([n['id'] for n in init_graph['nodes']])
            print(f"Num nodes in here: {len(graphs[0]['nodes'])}")
            print(f"Num nodes in init: {len(init_graph['nodes'])}")
            print(f"Ids in here but not in init: {len(ids_here - ids_init)}")
            print(f"Ids in init but not here: {len(ids_init - ids_here)}")
            print(f"Nodes in here but not in init: {len(id_classnames_here - id_classnames_init)}")
            print(f"Nodes in init but not here: {len(id_classnames_init - id_classnames_here)}")
            self.constant_nodes = [n for n in init_graph['nodes'] if n['id'] not in ids_here]
            constant_node_ids = [n['id'] for n in self.constant_nodes]
            self.constant_edges = [e for e in init_graph['edges'] if (e['from_id'] in constant_node_ids or e['to_id'] in constant_node_ids)]
        first_graph = deepcopy(graphs[0])
        first_graph['nodes'] = first_graph['nodes'] + self.constant_nodes
        first_graph['edges'] = first_graph['edges'] + self.constant_edges
        first_graph['nodes'].sort(key=lambda x: x['id'])
        # print(f"Num nodes in first graph: {len(first_graph['nodes'])}")
        temporary_data['node_idx_from_id'] = {}
        temporary_data['node_ids'] = []
        temporary_data['node_class_from_idx'] = []
        home_node = [{'id':-1, 'class_name':"home", 'category':"Home"}]
        
        nodes_in_graph = [node for node in first_graph['nodes'] if node['class_name'] not in self.common_data['ignored_node_classes']] + home_node
        exp_node_classes = [n['class_name'] for n in nodes_in_graph]
        assert exp_node_classes == self.common_data['node_classes'], f"exp_node_classes didn't match"
        exp_node_categories = [n['category'] for n in nodes_in_graph]
        assert exp_node_categories == self.common_data['node_categories'], f"exp_node_categories didn't match"
        exp_static_nodes = [n['id'] for n in nodes_in_graph if n['category'] in self.common_data['static_node_categories']]
        assert exp_static_nodes == self.common_data['static_nodes'], f"exp_static_nodes didn't match"
        expected_node_ids = [n['id'] for n in nodes_in_graph]
        assert expected_node_ids == self.common_data['node_ids'], f"expected_node_ids didn't match"
        
        # print(f"Num nodes in graph: {len(nodes_in_graph)}")
        num_nodes = len(nodes_in_graph)
        node_features = np.zeros((num_nodes))
        nonstatic_edges = torch.Tensor(1 - np.eye(num_nodes))
        for j,n in enumerate(nodes_in_graph):
            node_features[j] = self.get_node_index(n, j, temp_data=temporary_data)
            if n['category'] in self.common_data['static_node_categories']:
                nonstatic_edges[j,:] = 0
        edge_features = np.zeros((len(graphs), num_nodes, num_nodes))
        room_node_idxs = [temporary_data['node_idx_from_id'][n['id']] for n in nodes_in_graph if n['category']=="Rooms"]
        for i,graph in enumerate(graphs):
            graph['nodes'] = graph['nodes'] + self.constant_nodes
            graph['edges'] = graph['edges'] + self.constant_edges
            for room_node_index in room_node_idxs:
                edge_features[i,room_node_index, temporary_data['node_idx_from_id'][-1]] = 1
            for e in graph['edges']:
                if e['relation_type'] in self.common_data['edge_keys'] and e['from_id'] in temporary_data['node_ids'] and e['to_id'] in temporary_data['node_ids']:
                    edge_features[i,temporary_data['node_idx_from_id'][e['from_id']],temporary_data['node_idx_from_id'][e['to_id']]] = 1
            original_edges = edge_features[i,:,:]
            edge_features[i,:,:] = _sparsify(edge_features[i,:,:])
            edge_features[i, temporary_data['node_idx_from_id'][-1], temporary_data['node_idx_from_id'][-1]] = 1
            if (edge_features[i,:,:].sum(axis=-1)).max() != 1:
                # print(f"Matrix {i} not really a tree \n{edge_features[i,:,:]}")
                not_a_tree(original_edges, edge_features[i,:,:], temporary_data['node_class_from_idx'])
            # assert (edge_features[i,:,:].sum(axis=-1)).max() == 1, f"Matrix {i} not really a tree \n{edge_features[i,:,:]}"
        
        # print(f"Node features: {node_features.shape}")
        # print(f"Edge features: {edge_features.shape}")
        # print(f"Nonstatic edges: {nonstatic_edges.shape}")
        # # print(f"Node class from idx: {temporary_data['node_class_from_idx']}")
        # # print(f"Node ids: {temporary_data['node_ids']}")
        print()
        return torch.Tensor(node_features), torch.Tensor(edge_features), torch.Tensor(nonstatic_edges), temporary_data['node_class_from_idx'], temporary_data['node_ids']


    def stack_routine(self, edges, activities, times):
        ## TODO Maithili: Remove this for new datasets!!!
        times = torch.cat([times, torch.Tensor([float("Inf")])], dim=-1)

        data_idx = -1
        all_edges = []
        all_times = []
        all_activities = []
        for t in range(self.common_data['start_time'], self.common_data['end_time']+1, self.common_data['dt']):
            # if data_idx < 0:
                # data_idx += 1
                # continue
            while t >= times[data_idx+1]:
                data_idx += 1
            # if data_idx >= len(activities): data_idx = len(activities)-1
            all_edges.append(edges[data_idx].unsqueeze(0))
            all_times.append(torch.Tensor([t]))
            all_activities.append((activities[data_idx]).unsqueeze(0))
        stacked_edges = torch.cat(all_edges)
        stacked_activities = torch.cat(all_activities)
        stacked_times = torch.cat(all_times)
        movement_count = (stacked_edges[1:,:,:]-stacked_edges[:-1,:,:]).max(-1).values.sum(-1)
        # print(f'{movement_count.sum()}: {movement_count} movements!!')

        return stacked_edges, stacked_activities, stacked_times
 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model on routines.')
    parser.add_argument('--path', type=str, default='HouseholdA', help='Path where the data lives. Must contain routines, info and classes json files.')
    parser.add_argument('--dt', type=int, help='Custom timestep')
    parser.add_argument('-y', action='store_true', help='Overwrite if path exists.')
    args = parser.parse_args()

    output_dirname = 'processed_seqLM_coarse'

    ProcessDataset(args.path, output_path=os.path.join(args.path, output_dirname), overwrite=args.y, dt=args.dt)