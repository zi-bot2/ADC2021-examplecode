import argparse
import h5py
from sklearn.model_selection import train_test_split

def create_datasets_dense(bkg_file, signals_file, signal_data_names, events, test_size=0.2, val_size=0.2, input_shape=57):
    
    # read BACKGROUND data
    with h5py.File(bkg_file, 'r') as file:
        full_data = h5f['Particles'][:,:,:-1]
        np.random.shuffle(data)
        data = data[:events,:,:]
    
    # define training, test and validation datasets
    X_train, X_test = train_test_split(full_data, test_size=test_size, shuffle=True)
    X_train, X_val = train_test_split(X_train, test_size=val_size)

    del full_data
    
    # flatten the data for model input
    X_train = X_train.reshape(X_train.shape[0], input_shape)
    X_test = X_test.reshape(X_test.shape[0], input_shape)
    X_val = X_val.reshape(X_val.shape[0], input_shape)
    
    with h5py.File(bkg_file + '_dataset.h5', 'w') as h5f:
        h5f.create_dataset('X_train', data = X_train)
        h5f.create_dataset('X_test', data = X_test)
        h5f.create_dataset('X_val', data = X_val)
    
    if signal_file:
        # read SIGNAL data
        file2 = h5py.File(signals_file,'r')
        for signal_data_name in signal_data_names:
            sig_data_type = file2['%s' %(signal_data_name)][:]
            sig_data_type = sig_data_type.reshape(sig_data_type.shape[0],input_shape)
            with h5py.File(signal_data_name + '_dataset.h5', 'w') as h5f2:
                h5f2.create_dataset('%s'% (signal_data_name), data = sig_data_type)    
        file2.close()
        
    return

def create_datasets_convolutional(bkg_file, signals_file, signal_data_names, events, test_size=0.2, val_size=0.2):
    
    # read BACKGROUND data
    with h5py.File(bkg_file, 'r') as file:
        full_data = h5f['Particles'][:,:,:-1]
        np.random.shuffle(data)
        data = data[:events,:,:]
    
    # define training, test and validation datasets
    X_train, X_test = train_test_split(full_data, test_size=test_size, shuffle=True)
    X_train, X_val = train_test_split(X_train, test_size=val_size)

    del full_data
    
    with h5py.File(bkg_file + '_dataset.h5', 'w') as h5f:
        h5f.create_dataset('X_train', data = X_train)
        h5f.create_dataset('X_test', data = X_test)
        h5f.create_dataset('X_val', data = X_val)
    
    if signal_file:
        # read SIGNAL data
        file2 = h5py.File(signals_file,'r')
        for signal_data_name in signal_data_names:
            sig_data_type = file2['%s' %(signal_data_name)][:]
            sig_data_type = sig_data_type.reshape(sig_data_type.shape[0],input_shape)
            with h5py.File(signal_data_name + '_dataset.h5', 'w') as h5f2:
                h5f2.create_dataset('%s'% (signal_data_name), data = sig_data_type)    
        file2.close()
        
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bkg-file', type=str)
    parser.add_argument('--signals-file', type=str)
    parser.add_argument('--signal-names', type=str, action='append')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--val-size', type=float, default=0.2)
    parser.add_argument('--input-shape', type=int, default=57)
    args = parser.parse_args()
    create_datasets_dense(**vars(args))
