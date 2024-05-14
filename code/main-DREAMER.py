from cross_validation import *
# from prepare_data_DREAMER import *
from prepare_data_DREAMER2 import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ######## Data ########
    parser.add_argument('--dataset', type=str, default='DREAMER')
    parser.add_argument('--data-path', type=str, default='..\\..\\PyDreamer\\')
    parser.add_argument('--subjects', type=int, default=23) # 23 person each 18 clips
    parser.add_argument('--clips', type=int, default=18) # 23 person each 18 clips
    parser.add_argument('--num-class', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--label-type', type=str, default='A', choices=['A', 'V','D'])
    parser.add_argument('--segment', type=int, default=4, help='segment length in seconds')
    parser.add_argument('--trial-duration', type=int, default=60, help='trial duration in seconds')
    parser.add_argument('--overlap', type=float, default=0)
    parser.add_argument('--sampling-rate', type=int, default=128)
    parser.add_argument('--input-shape', type=tuple, default=(1, 14, 25472)) # 14 channel, 25472 datapoint
    parser.add_argument('--data-format', type=str, default='raw')
    ######## Training Process ########
    parser.add_argument('--random-seed', type=int, default=2021)
    parser.add_argument('--max-epoch', type=int, default=32)  
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--save-path', default='./save/')
    parser.add_argument('--load-path', default='./save/max-acc.pth')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--save-model', type=bool, default=True)
    ######## Model Parameters ########
    parser.add_argument('--model', type=str, default='TSception')
    parser.add_argument('--T', type=int, default=15)
    parser.add_argument('--graph-type', type=str, default='TS', choices=['TS', 'O'], 
                        help='TS for the channel order of TSception, O for the original channel order')
    parser.add_argument('--hidden', type=int, default=32)

    ######## Reproduce the result using the saved model ######
    parser.add_argument('--reproduce', action='store_true')
    args = parser.parse_args()

    sub_to_run = np.arange(args.subjects)
    clip_to_run = np.arange(args.clips)
    # print(sub_to_run)
    # print(clip_to_run)

    
    pd = PrepareData(args)
    # pd.run((sub_to_run,clip_to_run), split=True, feature=False, expand=True)
    # Repair Split functiom in prepare_data_DREAMER.py
    pd.run((sub_to_run,clip_to_run), split=True, feature=False, expand=True)

    cv = CrossValidation(args)
    seed_all(args.random_seed)
    cv.n_fold_CV(subject=sub_to_run, fold=10, reproduce=args.reproduce)  # To do leave one trial out please set fold=40
