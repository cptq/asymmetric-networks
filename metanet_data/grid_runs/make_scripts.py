import csv

lst = []

#MODEL_TYPE = "resnet8"
#MODEL_TYPE = "sparse_resnet8"
#MODEL_TYPE = "sigma_asym_resnet8"
MODEL_TYPE = "smaller_resnet8"


with open('hparams.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(reader):
        if i == 0:
            assert row[0] == 'lr'
            assert row[1] == 'weight_decay'
            assert row[2] == 'label_smoothing'
            assert row[3] == 'epochs'
            print(row)
        else:
            lr = float(row[0])
            weight_decay = float(row[1])
            label_smoothing = float(row[2])
            epochs = int(row[3])
            save_name = i-1

            command = f'python train_cifar.py --config-file default_config.yaml --training.lr {lr} --training.weight_decay {weight_decay} --training.label_smoothing {label_smoothing} --training.epochs {epochs} --model.model_type {MODEL_TYPE} --misc.save_path data/{MODEL_TYPE}/ --misc.save_name {save_name} \n'
            lst.append(command)

lst[-1] = lst[-1].rstrip('\n') # gets rid of final newline

f = open(f'{MODEL_TYPE}_dataset_train.sh', 'w')
f.writelines(lst)
f.close()
