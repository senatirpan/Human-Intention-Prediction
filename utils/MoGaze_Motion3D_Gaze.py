from torch.utils.data import Dataset
import numpy as np
import os


class MoGaze_Motion3D_Gaze(Dataset):

    def __init__(self, data_dir, subjects, input_n, output_n, actions='all', sample_rate=1):
        actions = self.define_actions(actions)
        self.sample_rate = sample_rate
        
        pose_gaze, dim_used = self.load_data(data_dir, subjects, input_n, output_n, actions)

        self.pose_gaze = pose_gaze
        self.dim_used = dim_used

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
        pose_gaze = []        
        for subj in subjects:
            path = data_dir + "/" + subj + "/"
            file_names = sorted(os.listdir(path))
            pose_xyz_file_names = {}
            gaze_file_names = {}
            for action_idx in np.arange(action_number):
                pose_xyz_file_names[actions[ action_idx ]] = []    
                gaze_file_names[actions[ action_idx ]] = []    
            for name in file_names:
                name_split = name.split('_')
                action = name_split[0]
                if action in actions:
                    data_type = name_split[-1][:-4]
                    if(data_type == 'xyz'):
                        pose_xyz_file_names[action].append(name)
                        #print("action: {}, file_name: {}".format(action, name))
                    if(data_type == 'head'):
                        gaze_file_names[action].append(name)
                        #print("action: {}, file_name: {}".format(action, name))
                
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
                        continue
                        #raise( ValueError, "sequence length {} is larger than frame number {}".format(seq_len, num_frames))

                    gaze_data_path = path + gaze_file_names[action][i]
                    gaze_data = np.load(gaze_data_path)
                    #print("Gaze shape: {}".format(gaze_data.shape))
                    pose_gaze_data = np.concatenate((pose_xyz_data, gaze_data), axis=1)
                            
                    fs = np.arange(0, num_frames - seq_len + 1)
                    fs_sel = fs
                    for i in np.arange(seq_len - 1):
                        fs_sel = np.vstack((fs_sel, fs + i + 1))
                    fs_sel = fs_sel.transpose()
                    #print(fs_sel)
                    seq_sel = pose_gaze_data[fs_sel, :]
                    seq_sel = seq_sel[0:-1:self.sample_rate, :, :]
                    #print(seq_sel.shape)
                    if len(pose_gaze) == 0:
                        pose_gaze = seq_sel
                    else:
                        pose_gaze = np.concatenate((pose_gaze, seq_sel), axis=0)
        
        joints_used = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        dim_used = np.sort(np.concatenate((joints_used * 3, joints_used * 3 + 1, joints_used * 3 + 2)))
        #print("Data size: {}".format(pose_gaze.shape))
        #print("Dimensions used: {}".format(dim_used))
        return pose_gaze, dim_used
        
  
    def __len__(self):
        return np.shape(self.pose_gaze)[0]

    def __getitem__(self, item):
        return self.pose_gaze[item]            

                
if __name__ == "__main__":
    data_dir = "/scratch/hu/pose_forecast/mogaze/"        
    input_n = 50
    output_n = 25
    actions = "pick"
    train_subjects = ['p1_1', 'p1_2', 'p2_1', 'p4_1', 'p5_1', 'p6_1', 'p6_2']
    test_subjects = ['p7_1', 'p7_2', 'p7_3']
    train_dataset = MoGaze_Motion3D_Gaze(data_dir, train_subjects, input_n, output_n, actions)
    print("Training data size: {}, Dimensions used: {}".format(train_dataset.pose_gaze.shape, train_dataset.dim_used))
    test_dataset = MoGaze_Motion3D_Gaze(data_dir, test_subjects, input_n, output_n, actions)
    print("Test data size: {}, Dimensions used: {}".format(test_dataset.pose_gaze.shape, test_dataset.dim_used))