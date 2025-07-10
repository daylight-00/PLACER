import sys,os
import random
import torch
import torch.distributed as dist
import itertools
import re
import json
import numpy as np
import pandas as pd
import scipy
import scipy.spatial
import networkx as nx
from openbabel import openbabel

sys.path.append(os.path.dirname(__file__))
import cifutils
import obutils


class PDBDataset():

    def __init__(self, 
                 csv : str,
                 ncpu : int,
                 world_size : int,
                 rank : int,
                 params : dict):
        
        # parse .csv
        df = pd.read_csv(csv)

        # remove bad structures
        sel = (df.num_heavy>=40)
        df = df[sel]
        N = df.shape[0]

        # sample from the set inversely proportional to the number of 1-hop neighbors
        weights = 1/(1+df.degree)
        total_size = int(np.floor(weights.sum()))
        sampler = DistributedWeightedSampler(weights = weights,
                                             total_size = total_size,
                                             world_size = world_size,
                                             rank = rank)

        #self.dataset = torch.utils.data.DataLoader(Dataset(df.label.to_list(), params),
        self.dataset = torch.utils.data.DataLoader(Dataset(df, params),
                                                   sampler=sampler,
                                                   num_workers=ncpu,
                                                   batch_size=1,
                                                   pin_memory=True,
                                                   collate_fn=lambda sample : sample)

    
class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataframe, params):
        
        self.df = dataframe
        self.labels = dataframe.label.tolist()
        self.params = params
        
        self.dims1d = (7, # group
                       18, # period
                       1, # is lanthanide
                       1, # is actinide
                       15, # group with lanthanides
                       15, # group with actinides
                       params['maxcharge']*2+1, # formal charge
                       params['maxhydr'], # number of hydrogens
                       params['maxhyb'], # hybridization state
                       1) # corrupted mask

        
        self.dims2d = (2,2,4,params['maxpath']+1)
        
        self.Parser = cifutils.CIFParser(skip_res=params['skip_res'])

        # parse the periodic table
        DIR = os.path.dirname(__file__)
        with open(f'{DIR}/../data/PeriodicTableJSON.json') as file:
            data = json.load(file)['elements']
        self.elements = {
            d['number']:{
                "period":d["period"],
                "group":d["group"],
                "is_lanthanide":d["category"]=="lanthanide",
                "is_actinide":d["category"]=="actinide",
                } for d in data}


    def OneHotF1D(self, 
                  f1d : torch.tensor,
                  qmask : torch.tensor = None,
                  hmask : torch.tensor = None) -> torch.tensor:
        '''one-hot-encode 1D features'''

        element,charge,nhyd,hyb = f1d.T

        if qmask is None: qmask = torch.zeros_like(charge, dtype=bool)
        qmask = qmask[:,None]
        
        if hmask is None: hmask = torch.zeros_like(nhyd,dtype=bool)
        hmask = hmask[:,None]

        # element embeddings
        element[element>118] = 0
        element = torch.nn.functional.one_hot(element,118+1)
        
        # charge embedding
        qmax = self.params['maxcharge']
        charge = torch.clamp(charge,min=-qmax,max=qmax)
        emb = torch.block_diag(torch.ones((qmax,qmax)).triu(),
                               torch.ones((1,1)),
                               torch.ones((qmax,qmax)).tril())
        charge = emb[charge+qmax]*(~qmask)

        # number of hydrogens
        hmax = self.params['maxhydr']
        nhyd = torch.ones((hmax+1,hmax)).tril(diagonal=-1)[nhyd]
        nhyd = nhyd*(~hmask)

        # hybridization
        hmax = self.params['maxhyb']
        hyb = torch.eye(hmax)[torch.clamp(hyb,min=0,max=hmax-1)]*(hyb<hmax)[:,None] # zero out if off-range
        
        # stack all features together
        f1d = torch.cat([element,charge,qmask,nhyd,hmask,hyb], dim=-1)
        
        return f1d


    def OneHotF1D_new(self, 
                  f1d : torch.tensor) -> torch.tensor:
        '''one-hot-encode 1D features'''

        # element embeddings
        element = [
            torch.nn.functional.one_hot(f1d[:,0],self.dims1d[0]+1)[:,1:],
            torch.nn.functional.one_hot(f1d[:,1],self.dims1d[1]+1)[:,1:],
            f1d[:,2:3],
            f1d[:,3:4],
            torch.nn.functional.one_hot(f1d[:,4],self.dims1d[4]+1)[:,1:],
            torch.nn.functional.one_hot(f1d[:,5],self.dims1d[5]+1)[:,1:] ]

        charge,nhyd,hyb = f1d[:,-3:].T

        # charge embedding
        qmax = self.params['maxcharge']
        charge = torch.clamp(charge,min=-qmax,max=qmax)
        emb = torch.block_diag(torch.ones((qmax,qmax)).triu(),
                               torch.ones((1,1)),
                               torch.ones((qmax,qmax)).tril())
        charge = emb[charge+qmax]

        # number of hydrogens
        hmax = self.params['maxhydr']
        nhyd = torch.ones((hmax+1,hmax)).tril(diagonal=-1)[nhyd]

        # hybridization
        hmax = self.params['maxhyb']
        hyb = torch.eye(hmax)[torch.clamp(hyb,min=0,max=hmax-1)]*(hyb<hmax)[:,None] # zero out if off-range
        
        # stack all features together
        f1d = torch.cat(element + [charge,nhyd,hyb], dim=-1)
        
        return f1d

    
    def OneHotF2D(self, f2d : torch.tensor) -> torch.tensor:
        '''one-hot-encode 2D features'''
    
        f2d = torch.cat([torch.nn.functional.one_hot(f,d)
                         for f,d in zip(f2d.permute(2,0,1),self.dims2d)], dim=-1).float()
        
        return f2d


    def get_crop_center(self, chains, skip_chains=None, exclude=None, multicenter=False):
        """
        Parameters
        ----------
        chains : dict
            Dictionary of parsed chains.
        skip_chains : list, optional
            Chains that will be excluded from being selected as corruption center. The default is None.
        exclude : list, optional
            Residues that will be excluded from being selected as crop center. The default is None.
            Format: [(chain, name3, resno), ...]
        multicenter : bool
            Whether multiple corruption centers should be picked, respecting any excluded residues.

        Returns
        -------
        center : cifutils.Atom
            DESCRIPTION.
        """

        # collect all physically observed atoms and count
        # occurencies of different molecule types
        # 0 - protein
        # 1 - NA
        # 2 - small molecules or a non-standard residue
        # 3 - metal
        atoms,types = [],[]
        for k,v in chains.items():
            if skip_chains is not None and k in skip_chains:
                continue
            for a in v.atoms.values():
                if a.occ==0 or a.element<=1:
                    continue
                if exclude is not None and (k, a.name[2], int(a.name[1])) in exclude:
                    continue
                # skipping backbone atoms
                if "polypeptide" in v.type and a.name[3] in self.params["bb_prot"]:
                    continue

                atoms.append(a)
                
                if 'nonpoly' in v.type or a.hetero==True:
                    if a.metal==True:
                        types.append(3)
                    else:
                        types.append(2)
                elif 'polyribo' in v.type or 'polydeoxyribo' in v.type:
                    types.append(1)
                else:
                    types.append(0)
        if len(atoms) == 0:
            print("WARNING: No atoms to select the center from. Either applicable ligands got excluded, or the atom properties (occ, element) are incompatible.")

        types = torch.tensor(types)
        types = torch.nn.functional.one_hot(types, num_classes=4)
        counts = types.sum(dim=0)

        # renormalize sampling probabilities based on atom counts
        p = torch.tensor(self.params['crop_prob']).float()
        norm = p[counts>0].sum()
        if norm>1e-3:
            p = p/norm
            p = (types*p/counts)[:,counts>0].sum(dim=1).numpy().astype('float64')
            p /= p.sum()
        else:
            N = types.shape[0]
            p = np.array([1./N]*N)
            p /= p.sum()

        # samples an atom around which to crop
        if multicenter is False:
            center = [ atoms[np.random.choice(len(atoms), p=p)] ]
        else:
            # split atoms to residues
            # at this poiunt all atoms should already be valid for multicenter prediction
            _residues = list(set([a.name[:3] for a in atoms]))
            atoms_split = {_res: [] for _res in _residues}
            p_split = {_res: [] for _res in _residues}
            for a,_p in zip(atoms,p):
                atoms_split[a.name[:3]].append(a)
                p_split[a.name[:3]].append(_p)

            center = []
            for _res,_atoms in atoms_split.items():
                _p = np.array(p_split[_res])*(1/sum(p_split[_res]))  # normalizing probabilities to 1
                center.append( _atoms[np.random.choice(len(_atoms), p=_p)] )

        return center

    
    def get_crop(self, chains, center, center_exclude_chains=None):

        # collect all atoms observed in the structure
        atoms = [a for k,v in chains.items() for a in v.atoms.values() if a.occ>0 and a.element>1]
        
        # find neighboring atoms
        kd = scipy.spatial.cKDTree(data=[a.xyz for a in atoms])
        maxatoms = len(atoms)
        if self.params['maxatoms']<maxatoms:
            maxatoms = self.params['maxatoms']

        if isinstance(center[0], (float, np.float32, np.float64)):
            # center is one point
            dist,indices = kd.query(center,maxatoms)
            if center_exclude_chains is not None:
                indices = [j for j in indices if atoms[j].name[0] not in center_exclude_chains]
        else:
            center_coords = [cntr.xyz if hasattr(cntr, "xyz") else cntr for cntr in center]
            # center is a collection of points
            matches = [(i,d) for p in center_coords for d,i in zip(*kd.query(p,maxatoms))]
            matches = sorted(matches,key=lambda x : x[0])
            matches = [(i,sorted([d for j,d in g])[0]) for i,g in itertools.groupby(matches, key=lambda x : x[0])]
            matches = sorted(matches,key=lambda x : x[1])[:maxatoms]
            #dist = [d for i,d in matches]
            indices = [i for i,d in matches]

        # collect all residues which belong to the crop
        cropped_dict = dict.fromkeys([atoms[i].name[:3] for i in indices],0)
        atoms = [a for k,v in chains.items() for a in v.atoms.values() if a.element!=1]
        for a in atoms:
            if a.name[:3] in cropped_dict.keys():
                cropped_dict[a.name[:3]] += 1
        
        res = list(cropped_dict.keys())
        naa = np.array(list(cropped_dict.values()))
        nres = sum(naa.cumsum()<self.params['maxatoms'])
        cropped_res = set([tuple(r) for r in res[:nres]])
        
        atoms = [a for a in atoms if a.name[:3] in cropped_res]
        
        return atoms


    def get_crop_around_mol(self, chains, obmol, exclude=None, multicenter=False):
        """
        Parameters
        ----------
        chains : dict
            Dictionary of parsed chains.
        obmol : openbabel.OBMol
            Ligands openbabel object
        exclude : list, optional
            Residues that will be excluded from being selected as crop center. The default is None.
            Format: [(chain, name3, resno), ...]
        multicenter : bool
            Whether multiple corruption centers should be picked, respecting any excluded residues.

        Returns
        -------
        atoms : list
            Cropped atoms list
        center : cifutils.Atom
            DESCRIPTION.
        """
        # extract coords from the ligand
        kd_lig = scipy.spatial.cKDTree(
            data=[(a.x(),a.y(),a.z()) for a in openbabel.OBMolAtomIter(obmol)]
        )

        # collect all atoms observed in the structure
        atoms = [a for k,v in chains.items() for a in v.atoms.values() if a.occ>0 and a.element>1]
        kd = scipy.spatial.cKDTree(data=[a.xyz for a in atoms])

        # pick a center
        if multicenter is False:
            indices = kd_lig.query_ball_tree(kd,r=0.5)
            indices = list({j for i in indices for j in i})
            if exclude is not None:
                indices = [j for j in indices if (atoms[j].name[0], atoms[j].name[2], int(atoms[j].name[1])) not in exclude]
            center = [atoms[random.choice(indices)]]
        else:
            center = []
            # Picking a center for each ligand residue separately
            for obRes in openbabel.OBResidueIter(obmol):
                if (obRes.GetChain(), obRes.GetName(), obRes.GetNum()) in exclude:
                    continue
                kd_lig = scipy.spatial.cKDTree(
                    data=[(a.x(),a.y(),a.z()) for a in openbabel.OBResidueAtomIter(obRes)])
                indices = kd_lig.query_ball_tree(kd,r=0.5)
                indices = list({j for i in indices for j in i})
                center.append(atoms[random.choice(indices)])
            
            # Resetting the ligand tree to include all residues
            kd_lig = scipy.spatial.cKDTree(
                data=[(a.x(),a.y(),a.z()) for a in openbabel.OBMolAtomIter(obmol)])


        # find neighboring atoms
        maxatoms = len(atoms)
        if self.params['maxatoms']<maxatoms:
            maxatoms = self.params['maxatoms']
        #dist,indices = kd.query(center.xyz,maxatoms)

        d = scipy.spatial.distance.cdist(kd.data,kd_lig.data)
        i = d.flatten().argsort()
        indices = np.unravel_index(range(i.shape[0]), d.shape)[0][i]
        indices = list(dict.fromkeys(indices))[:maxatoms]

        # collect all residues which belong to the crop
        cropped_dict = dict.fromkeys([atoms[i].name[:3] for i in indices],0)
        atoms = [a for k,v in chains.items() for a in v.atoms.values() if a.element!=1]
        for a in atoms:
            if a.name[:3] in cropped_dict.keys():
                cropped_dict[a.name[:3]] += 1
        
        mask = np.array(list(cropped_dict.values())).cumsum()<maxatoms
        cropped_res = {r for r,flag in zip(cropped_dict.keys(),mask) if flag==True}
        
        atoms = [a for a in atoms if a.name[:3] in cropped_res]
        
        return atoms,center

    
    def get_atom_graph(self, chains, covale, crop):

        # collect bonds within the crop
        atom_names = set([a.name for a in crop])
        bonds = [bond for k,v in chains.items() for bond in v.bonds+covale
                 if bond.a in atom_names and bond.b in atom_names]

        # build graph
        G = nx.Graph()
        G.add_nodes_from([(a.name,{'Y':torch.tensor(a.xyz),'index':i,'atom':a})
                          for i,a in enumerate(crop)])
        G.add_edges_from([(bond.a,bond.b,{'bond':bond}) for bond in bonds])

        # add node attributes to the graph
        nx.set_node_attributes(G,False,'has_xyz')
        nx.set_node_attributes(G.subgraph([a.name for a in crop if a.occ>0.0]),True,'has_xyz')

        bb_atoms = []
        for a in crop:
            ctype = chains[a.name[0]].type
            if ('polyribo' in ctype or 'polydeoxyribo' in ctype) and a.name[-1] in self.params['bb_na']:
                bb_atoms.append(a.name)
            elif 'polypept' in ctype and a.name[-1] in self.params['bb_prot']:
                bb_atoms.append(a.name)

        nx.set_node_attributes(G,False,'is_bb')
        nx.set_node_attributes(G.subgraph(bb_atoms),True,'is_bb')
        nx.set_node_attributes(G,False,'corrupted')

        return G

    
    def map_to_anchors(self, G):

        cutoff = self.params['maskrad']
        
        # collect backbone atoms with 3D coords
        anchors = {n[0] for n in G.nodes(data=True) if n[1]['is_bb']==True and n[1]['has_xyz']==True}

        # map the remaining atoms to closest anchors
        pairs = []
        to_perturb = []
        for cc in nx.connected_components(G):
            
            # if the graph has connected componets w/o anchors
            # then randomly sample additional anchors
            if cc.isdisjoint(anchors):
                H = G.subgraph(cc)
                
                # mark all atoms in H as corrupted
                nx.set_node_attributes(H,True,'corrupted')
                
                observed_atoms = [n[0] for n in H.nodes(data=True) if n[1]['has_xyz']==True]
                random.shuffle(observed_atoms)
                
                # for a disconnected component H w/o anchors, add new anchors
                # propotrionally to the diameter of of H
                diam = nx.diameter(H)
                anchors_i = observed_atoms[:max(1,diam//(2*cutoff))]
                pairs.extend([next(((i,j) for j,_ in nx.single_target_shortest_path_length(H,i).items() if j in anchors_i)) for i in cc])
                to_perturb.extend(anchors_i)
            
            else:
                # TODO: account for big patches of structure w/o bb
                pairs.extend([next(((i,j) for j,_ in nx.single_target_shortest_path_length(G,i).items() if j in anchors)) for i in cc])
        
        return pairs, to_perturb

    
    @staticmethod
    def find_all_paths_of_length_n(G : nx.Graph,
                                   n : int) -> torch.Tensor:
        '''find all paths of length N in nx.Graph
        https://stackoverflow.com/questions/28095646/finding-all-paths-walks-of-given-length-in-a-networkx-graph'''

        def findPaths(G,u,n):
            if n==0:
                return [[u]]
            paths = [[u]+path for neighbor in G.neighbors(u) for path in findPaths(G,neighbor,n-1) if u not in path]
            return paths

        # all paths of length n
        allpaths = [[G.nodes[pi]['index'] for pi in p] for node in G for p in findPaths(G,node,n)]
        allpaths = [tuple(p) if p[0]<p[-1] else tuple(reversed(p)) for p in allpaths]

        # unique paths
        allpaths = list(set(allpaths))
        allpaths.sort()

        return allpaths


    @staticmethod
    def get_topology(chains, G):
        
        bonds = [(G.nodes[i]['index'],G.nodes[j]['index']) for i,j in G.edges]
        bondlen = [attr['bond'].length for i,j,attr in G.edges(data=True)]
        angles = Dataset.find_all_paths_of_length_n(G,2)
        torsions = Dataset.find_all_paths_of_length_n(G,3)
        
        # select automorphisms belonging to the crop
        to_set = lambda a : set([aij for ai in a for aij in ai])
        automorphisms = [a for c in chains.values() for a in c.automorphisms if to_set(a).issubset(G.nodes)]
        
        # don't consider these atoms when selecting frames for FAPE calculation
        skip_from_frames = set(G.nodes[aij]['index'] for a in automorphisms for aij in to_set(a))
        frames = [a for a in angles if set(a).isdisjoint(skip_from_frames)]

        # subselect further to only include observed atoms
        observed = {node for node,attr in G.nodes(data=True) if attr['atom'].occ>0.0}
        automorphisms = [a for a in automorphisms if to_set(a).issubset(observed)]
        automorphisms = [torch.tensor([[G.nodes[aij]['index'] for aij in ai] for ai in a]) for a in automorphisms]
        
        # chirals and planars
        chirals = [[G.nodes[i]['index'] for i in chi] for c in chains.values() 
                   for chi in c.chirals if set(chi).issubset(G.nodes)]
        planars = [[G.nodes[i]['index'] for i in chi] for c in chains.values() 
                   for chi in c.planars if set(chi).issubset(G.nodes)]
        
        return {
            'bonds' : torch.tensor(bonds),
            'bondlen' : torch.tensor(bondlen),
            'angles' : torch.tensor(angles),
            'torsions' : torch.tensor(torsions),
            'frames' : torch.tensor(frames),
            'chirals' : torch.tensor(chirals),
            'planars' : torch.tensor(planars),
            'permuts' : automorphisms
        }


    def get_features_new(self, G):
        
        N = len(G)
        
        ##### 1D #####
        atoms = [attr['atom'] for node,attr in G.nodes(data=True)]
        
        def get_atom_features(a):
            element = self.elements.get(a.element)
            if element is not None:
                element = self.elements[a.element]
                period,group,is_lan,is_act = element.values()
                group_lan = ((a.element-56)%16)*is_lan
                group_act = ((a.element-88)%16)*is_act
                return (period, # element's period
                        group, # element's period
                        is_lan, # is the element a lanthanide?
                        is_act, # is the element an actinide?
                        group_lan, # element's group within lanthanides
                        group_act, # element's group within actinides
                        a.charge, # element's charge
                        a.nhyd, # element's number of hydrogens
                        a.hyb) # element's hybridization
            else:
                return (0,0,0,0,0,0,0,0,0)
        
        f1d = torch.tensor([get_atom_features(a) for a in atoms])
        
        ##### 2D #####
        
        # aromatic, in ring, order
        a2i = {n:attr['index'] for n,attr in G.nodes(data=True)}
        i,j = np.array([(a2i[i],a2i[j]) for i,j in G.edges]).T
        bonds = [attr['bond'] for i,j,attr in G.edges(data=True)]
        f2d = torch.zeros((N,N,4), dtype=torch.long)
        f2d[i,j,:3] = torch.tensor([(b.aromatic,b.in_ring,b.order) for b in bonds])
        f2d[j,i,:3] = f2d[i,j,:3]

        # bond separation
        paths = dict(nx.all_pairs_shortest_path_length(G,cutoff=self.params['maxpath']))
        paths = [(a2i[i],a2i[j],vij) for i,vi in paths.items() for j,vij in vi.items()]
        i,j,v = torch.tensor(paths).T
        f2d[i,j,3] = v
        
        return f1d,f2d
    
        
    def __len__(self):
        return len(self.labels)

    
    def __getitem__(self, index):
        """
        Dataloader iterator used during training.
        Parses mmCIF input from a defined storage location.
        Creates an up to 600 atom crop around a random atom.
        Identifies bonded atoms around that random atom for corruption.
        Corrupts coordinates around the random atom, and residues around their backbone.
        Computes 1D and 2D features.
        One-hot-encodes the 1D and 2D features.
        Returns a dictionary to be used as a network input during training.
        """

        #label = self.labels[index]
        label = self.df.iloc[index].label
        out = {'label' : label}

        # load .cif file
        fname = "%s/%s/%s.cif.gz"%(self.params['DIR'], label[1:3], label)
        chains,asmb,covale,meta = self.Parser.parse(fname)

        if 'ligand' in self.df.columns:
            # crop around a given ligand
            fname = self.df.iloc[index].ligand
            obmol = openbabel.OBMol()
            obConversion = openbabel.OBConversion()
            if fname.endswith('.mol2'):
                obConversion.SetInFormat("mol2")
            elif fname.endswith('.mol'):
                obConversion.SetInFormat("mol")
            else:
                sys.error('Error: wrong file type "%s"'%(fname))
            obConversion.ReadFile(obmol,fname)
            obmol.DeleteHydrogens()
            cropped_atoms,center = self.get_crop_around_mol(chains, obmol)
        else:
            # crop around a random atom
            center = self.get_crop_center(chains)
            cropped_atoms = self.get_crop(chains, center.xyz)

        # convert the crop to nx.Graph
        G = self.get_atom_graph(chains, covale, cropped_atoms)
        L = len(G)
        
        # corrupt bonded neighbourhood around the center
        cutoff = self.params['maskrad']
        H = G.subgraph(nx.single_source_shortest_path_length(G, center.name, cutoff=cutoff))
        nx.set_node_attributes(H,False,'is_bb')

        corrupted = {n[1]['index'] for n in G.nodes(data=True) if (n[0] in H.nodes) and (n[1]['atom'].occ>0.0)}
        corrupted.update({n[1]['index'] for n in G.nodes(data=True) if n[1]['corrupted']==True})
        
        corrupted = list(corrupted)
        corrupted.sort()
        out['corrupted'] = torch.tensor(corrupted)

        # map corrupted atoms to closest anchors
        pairs,to_perturb = self.map_to_anchors(G)

        # randomize starting coords:
        #  - perturb extra anchors
        for node in to_perturb:
            Y = G.nodes[node]['Y']
            G.nodes[node]['X0'] = Y + torch.randn_like(Y)*self.params['sigma']

        #  - perturb backbone
        for node in set(p[1] for p in pairs):
            Y = G.nodes[node]['Y']
            G.nodes[node]['X0'] = Y + torch.randn_like(Y)*self.params['sigma_bb']

        #  - initialize the rest based on closest anchors
        for node,anchor in pairs:
            X0 = G.nodes[anchor]['X0']
            if node==anchor:
                G.nodes[node]['X'] = X0
            else:
                G.nodes[node]['X'] = X0 + torch.randn_like(X0)*self.params['sigma']

        # get topology
        out.update(Dataset.get_topology(chains,G))
        
        # get input features
        f1d,f2d = self.get_features_new(G)
        separation = f2d[...,-1]
        qmask = torch.bernoulli(torch.full((L,),self.params['maskrate_q'])).bool()
        hmask = torch.bernoulli(torch.full((L,),self.params['maskrate_h'])).bool()
        f1d = self.OneHotF1D_new(f1d)
        f2d = self.OneHotF2D(f2d)
        crpt = torch.zeros((L,1))
        crpt[out['corrupted']] = 1
        f1d = torch.cat([f1d,crpt],dim=-1)
        out.update({
            'f1d' : f1d,
            'f2d' : f2d,
            'separation' : separation
        })
        
        # add coordinates and mask
        out.update({
            'X' : torch.stack([n[1]['X'] for n in G.nodes(data=True)]),
            'Y' : torch.stack([n[1]['Y'] for n in G.nodes(data=True)]),
            'observed' : torch.tensor([n[1]['atom'].occ>0 for n in G.nodes(data=True)]),
            'G' : G
        })
        
        return out
        

class DistributedWeightedSampler(torch.utils.data.Sampler):
    def __init__(self, weights, total_size, world_size, rank):

        self.weights = torch.tensor(weights).float()
        self.world_size = world_size
        self.rank = rank
        self.epoch = 0

        # adjust sample size to be devisible by the number of available gpus
        self.sample_size = total_size//world_size
        self.total_size = self.sample_size*world_size
        

    def __iter__(self):
        
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # sample indices
        indices = torch.multinomial(self.weights, self.total_size, 
                                    replacement=False, generator=g)

        # shuffle indices
        indices = indices[torch.randperm(self.total_size, generator=g)]

        # select according to rank
        indices = indices[self.rank::self.world_size]

        return iter(indices.tolist())


    def __len__(self):
        return self.sample_size


    def set_epoch(self, epoch):
        self.epoch = epoch
