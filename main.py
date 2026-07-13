import os
import argparse
from ChallengeDL.solver import Solver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=40)
    
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoint")
    parser.add_argument("--ckpt_name", type=str, default="depth")
    parser.add_argument("--evaluate_every", type=int, default=1)
    parser.add_argument("--visualize_every", type=int, default=50)
    parser.add_argument("--data_dir", type=str, default="../data/DepthEstimationUnreal")
    # default=os.path.join("C:\\", "Users", "Utente", "dataset", "DepthEstimationUnreal"))

    parser.add_argument("--is_train", type=bool, default=False)
    parser.add_argument("--ckpt_file", type=str, default="depth_34.pth")

    args = parser.parse_args()
    solver = Solver(args)
    if args.is_train:
        solver.fit()
    else:
        solver.test()

if __name__ == "__main__":
    main()
