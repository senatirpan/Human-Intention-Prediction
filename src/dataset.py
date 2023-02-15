from torch.utils.data import Dataset
import numpy as np
import os

class Pose_Gaze_Intention(Dataset):

    def __init__(self, data_dir, subjects, input_n, output_n, actions = 'all', sample_rate=1):
        actions = self.define_actions(actions)
        pose_gaze_intention = self.load_data(data_dir, subjects, input_n, output_n, actions)
        self.pose_gaze_intention = pose_gaze_intention
        self.sample_rate = sample_rate
        

    def define_actions(self, action):
        """
        Define the list of actions we are using.

        Args
        action: String with the passed action. Could be "all"
        Returns
        actions: List of strings of actions
        Raises
        ValueError if the action is not included.
        """

        actions = ["pick", "place"]

        if action in actions:
            return [action]
        if action == "all":
            return actions
        raise( ValueError, "Unrecognised action: %d" % action )
        
    def load_data(self, data_dir, subjects, input_n, output_n, actions):
        action_number = len(actions)
        seq_len = input_n + output_n
        pose_gaze_intention = []

        for subj in subjects:
            path = data_dir + "/" + subj + "/"
            file_names = sorted(os.listdir(path))  
            pose_xyz_file_names = {}
            gaze_file_names = {}
            intention_file_names = {}            
            
            for action_idx in np.arange(action_number):
                pose_xyz_file_names[actions[ action_idx ]] = []    
                gaze_file_names[actions[ action_idx ]] = []                
                intention_file_names[actions[ action_idx ]] = []                
            
            for name in file_names:
                name_split = name.split('_')
                action = name_split[0]
                if action in actions:
                    data_type = name_split[-1][:-4]
                    if(data_type == 'xyz'):
                        pose_xyz_file_names[action].append(name)
                if(data_type == 'gaze'):
                    gaze_file_names[action].append(name)                    
                if(data_type == 'intention'):
                    intention_file_names[action].append(name)
                
            for action_idx in np.arange(action_number):
                action = actions[ action_idx ]
                segments_number = len(pose_xyz_file_names[action])
                print("Reading subject {}, action {}, segments number {}".format(subj, action, segments_number))
                
                for i in range(segments_number):                                  
                    pose_xyz_data_path = path + pose_xyz_file_names[action][i]
                    pose_xyz_data = np.load(pose_xyz_data_path)
                    #print("Pose xyz shape: {}".format(pose_xyz_data.shape))                    
                    num_frames = pose_xyz_data.shape[0]
                    if num_frames < seq_len:
                        #raise( ValueError, "sequence length {} is larger than frame number {}".format(seq_len, num_frames))
                        continue

                    gaze_data_path = path + gaze_file_names[action][i]
                    gaze_data = np.load(gaze_data_path)

                    intention_data_path = path + intention_file_names[action][i]
                    intention_data = np.load(intention_data_path)[:, :3]

                    pose_gaze_intention_data = np.concatenate((np.concatenate((pose_xyz_data, gaze_data), axis=1), intention_data), axis=1)
                    #print('aaa:', pose_gaze_intention_data.shape) # aaa: (1591, 69)
                            
                    fs = np.arange(0, num_frames - seq_len + 1)

                    fs_sel = fs 
                    #print('aa:', fs_sel.shape) # aa: (1581,)
                    #print('aa_d:', fs_sel)
                    for i in np.arange(seq_len - 1):
                        fs_sel = np.vstack((fs_sel, fs + i + 1))
                    #print('bb:', fs_sel[1]) # bb: (11, 1581)
                    fs_sel = fs_sel.transpose()
                    #print('cc:', fs_sel.shape) # cc: (1581, 11)
                    seq_sel = pose_gaze_intention_data[fs_sel, :]
                    #print('a:', seq_sel.shape) # a: (1581, 11, 69)
                    datasize = seq_sel.shape[0] 
                    seq_sel = seq_sel[0:datasize:10, :, :]
                    if len(pose_gaze_intention) == 0:
                        pose_gaze_intention = seq_sel
                    else:
                        pose_gaze_intention = np.concatenate((pose_gaze_intention, seq_sel), axis=0)
        
        return pose_gaze_intention
        
  
    def __len__(self):
        return np.shape(self.pose_gaze_intention)[0]

    def __getitem__(self, item):
        return self.pose_gaze_intention[item]